from __future__ import annotations

import json

import cv2
import pytest

from dense_fixtures import generate_dense
from style_demo import Unit, enrich_vl_markdown, estimate_color, stroke_width


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
