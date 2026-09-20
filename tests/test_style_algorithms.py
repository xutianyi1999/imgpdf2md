from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from dense_fixtures import generate_dense
from style_demo import (
    Unit,
    classify_units,
    enrich_vl_markdown,
    estimate_color,
    stroke_width,
    textar_span_candidates,
    make_spans,
)
from textar_backend import TexTARBackend
from style_matrix_fixtures import CASES as STYLE_MATRIX_CASES


@pytest.fixture(scope="module")
def dense_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("dense-fixtures")
    manifest = json.loads(generate_dense(root).read_text(encoding="utf-8"))
    return root, manifest["cases"]


def test_accent_color_detection_on_dense_mobile_policy(dense_cases):
    root, cases = dense_cases
    case = next(item for item in cases if item["name"] == "dense_mobile_policy")
    image = cv2.imread(str(root / case["image"]))
    accents = [unit for unit in case["units"] if unit["colored"] and unit["text"].isalnum()]
    detected = [estimate_color(image, unit["box"])[1] for unit in accents]
    assert len(detected) >= 20
    assert sum(detected) / len(detected) >= 0.95


def test_bold_has_larger_average_stroke_width_in_dense_policy(dense_cases):
    root, cases = dense_cases
    case = next(item for item in cases if item["name"] == "dense_mobile_policy")
    image = cv2.imread(str(root / case["image"]))
    widths = {True: [], False: []}
    for unit in case["units"]:
        if not unit["text"].isalnum():
            continue
        x1, y1, x2, y2 = unit["box"]
        value = stroke_width(image[y1:y2, x1:x2])
        if value is not None:
            widths[bool(unit["bold"])].append(value)
    assert sum(widths[True]) / len(widths[True]) > sum(widths[False]) / len(widths[False])


def test_styles_are_aligned_back_to_structured_vl_markdown():
    text = "标题普通粗体蓝色"
    units = [Unit(char, [0, 0, 1, 1], 0) for char in text]
    for unit in units[4:6]:
        unit.bold = True
        unit.bold_confidence = 0.9
    for unit in units[6:]:
        unit.color = "#087cfb"
        unit.colored = True
        unit.color_confidence = 0.9
    rendered, coverage = enrich_vl_markdown("## 标题\n\n普通粗体蓝色", units)
    assert rendered.startswith("## 标题")
    assert "**粗体**" in rendered
    assert '<span style="color:#0080ff">蓝色</span>' in rendered
    assert coverage == 1.0


def test_dense_fixtures_cover_compact_serif_and_dark_documents(dense_cases):
    _, cases = dense_cases
    names = {case["name"] for case in cases}
    assert {
        "compact_mobile_consent",
        "dark_mode_privacy_policy",
        "inline_bold_boundary_stress",
    } <= names
    agreement = next(case for case in cases if case["name"] == "dense_user_service_agreement")
    assert len(agreement["units"]) >= 1000


def test_textar_preprocessing_normalizes_dark_mode_to_light_background():
    image = np.full((30, 30, 3), 24, np.uint8)
    cv2.line(image, (12, 7), (12, 22), (235, 235, 235), 2)
    tensor = TexTARBackend._tensor_crop(image, [5, 5, 20, 25])
    assert tensor.shape == (3, 128, 96)
    assert float(np.median(tensor[:, :, :10])) > 0.8


def test_textar_span_fills_weak_characters_and_rejects_isolated_spikes():
    units = [Unit(char, [index * 11, 0, index * 11 + 10, 18], 0) for index, char in enumerate("甲乙丙丁戊己庚辛壬")]
    probabilities = [0.82, 0.10, 0.35, 0.56, 0.35, 0.08, 0.76, 0.10, 0.10]
    for unit, probability in zip(units, probabilities):
        unit.textar = {"bold": probability}
    # The middle 0.35/0.56/0.35 run is coherent; isolated 0.82 and 0.76 are not.
    assert textar_span_candidates(units) == {2, 3, 4}


def test_textar_span_does_not_cross_punctuation_or_line_boundaries():
    units = [
        Unit("重", [0, 0, 10, 18], 0, textar={"bold": 0.60}),
        Unit("，", [11, 0, 21, 18], 0, textar={"bold": 0.60}),
        Unit("点", [22, 0, 32, 18], 0, textar={"bold": 0.60}),
        Unit("新", [0, 22, 10, 40], 1, textar={"bold": 0.60}),
    ]
    assert textar_span_candidates(units) == set()


def test_style_matrix_covers_font_size_weight_and_theme_axes():
    assert len(STYLE_MATRIX_CASES) >= 12
    assert {case[1] for case in STYLE_MATRIX_CASES} == {
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
    }
    assert {case[2] for case in STYLE_MATRIX_CASES} >= {12, 13, 14, 16, 20}
    assert {case[3] for case in STYLE_MATRIX_CASES} >= {600, 700, 900}
    assert {case[4] for case in STYLE_MATRIX_CASES} >= {"light", "warm", "gray", "dark"}


def test_accepted_contextual_bold_survives_markdown_rendering():
    class Predictions:
        def predict(self, image, units):
            return [{"bold": p} for p in (.35, .8, .35)]

    image = np.full((30, 45, 3), 255, np.uint8)
    units = [Unit(c, [i * 12, 0, i * 12 + 11, 18], 0) for i, c in enumerate("授权书")]
    classify_units(image, units, [[0, 0, 35, 18]], None, Predictions())
    assert make_spans(units)[1] == "**授权书**"
    classify_units(image, units, [[0, 0, 35, 18]], None, Predictions(), preserve_span_confidence=False)
    assert make_spans(units)[1] == "授**权**书"


def test_policy_fixture_has_true_negative_and_wrapped_table(tmp_path):
    from policy_fixtures import generate_policy

    manifest = json.loads(generate_policy(tmp_path).read_text())
    plain = next(c for c in manifest['cases'] if c['name'] == 'privacy_plain_negative')
    assert len(plain['units']) > 700
    assert not any(u['bold'] or u['underline'] or u['colored'] for u in plain['units'])
    table = next(c for c in manifest['cases'] if c['name'] == 'sdk_table_wrapped')
    assert sum(u['tag'] == 'td' for u in table['units']) > 100
