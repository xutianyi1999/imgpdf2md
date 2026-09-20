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
    assert "<strong>粗体</strong>" in rendered
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
    assert make_spans(units)[1] == "<strong>授权书</strong>"
    classify_units(image, units, [[0, 0, 35, 18]], None, Predictions(), preserve_span_confidence=False)
    assert make_spans(units)[1] == "授<strong>权</strong>书"


def test_policy_fixture_has_true_negative_and_wrapped_table(tmp_path):
    from policy_fixtures import generate_policy

    manifest = json.loads(generate_policy(tmp_path).read_text())
    plain = next(c for c in manifest['cases'] if c['name'] == 'privacy_plain_negative')
    assert len(plain['units']) > 700
    assert not any(u['bold'] or u['underline'] or u['colored'] for u in plain['units'])
    table = next(c for c in manifest['cases'] if c['name'] == 'sdk_table_wrapped')
    assert sum(u['tag'] == 'td' for u in table['units']) > 100


def test_word_context_respects_table_gaps_and_preserves_other_attributes():
    from bold_context import WordContextEnsemble, word_groups

    units = [Unit(c, [x, 0, x+10, 18], 0) for c,x in zip('隐私政策', (0,11,90,101))]
    groups, owners = word_groups(units)
    assert all(not (0 in indices and 2 in indices) for indices in owners)
    assert sorted(i for indices in owners for i in indices) == [0,1,2,3]
    _, mixed_owners = word_groups([Unit('SDK-v2',[0,0,60,18],0),Unit('授权',[61,0,83,18],0)])
    assert sorted(i for indices in mixed_owners for i in indices)==[0,1]

    class Backend:
        def predict(self, image, items):
            return [{'bold': .1 if len(u.text)==1 else .8, 'underline': .37} for u in items]

    result = WordContextEnsemble(Backend()).predict(np.zeros((20,120,3),np.uint8), units)
    assert all(p['bold']==.1 and p['underline']==.37 for p in result)
    assert all(p.get('word_bold')==.8 for p in result)


def test_word_context_requires_support_and_renders_accepted_words():
    from bold_context import accept_word_bold

    unsupported = Unit('字',[0,0,10,18],0,textar={'word_bold':.95,'character_bold':.01})
    assert not accept_word_bold(unsupported)

    class Backend:
        def predict(self,image,units):
            return [{'bold':.1,'word_bold':.7,'character_bold':.1,'word_context_enabled':1.} for _ in units]

    units=[Unit(c,[i*12,0,i*12+11,18],0) for i,c in enumerate('授权书')]
    classify_units(np.full((30,45,3),255,np.uint8),units,[[0,0,35,18]],None,Backend())
    assert make_spans(units)[1]=='<strong>授权书</strong>'


def test_bold_gap_closing_is_supported_bounded_and_nonrecursive():
    from bold_context import bridge_bold_gaps

    units=[Unit(c,[i*12,0,i*12+11,18],0,textar={'bold':.2}) for i,c in enumerate('甲乙丙丁戊己庚')]
    for i in (0,3,6):
        units[i].bold=True
        units[i].bold_confidence=.7
    assert bridge_bold_gaps(units)=={1,2,4,5}
    units[3].line_index=1
    assert bridge_bold_gaps(units)==set()
    units[3].line_index=0
    units[1].textar={'bold':.01}
    assert bridge_bold_gaps(units)=={4,5}
    units[4].box=[100,0,111,18]
    assert bridge_bold_gaps(units)==set()


def test_adjacent_bold_with_color_change_has_one_markdown_wrapper():
    units=[Unit(c,[i*12,0,i*12+11,18],0,bold=True,bold_confidence=.8) for i,c in enumerate('蓝色正文')]
    for u in units[:2]:
        u.colored=True
        u.color='#0080ff'
    expected='<strong><span style="color:#0080ff">蓝色</span>正文</strong>'
    assert make_spans(units)[1]==expected
    assert enrich_vl_markdown('蓝色正文',units)[0]==expected


def test_soft_wrap_requires_aligned_lines_tight_leading_and_supported_gap():
    from bold_context import wrapped_bold_candidates

    units=[Unit('字',[col*18,row*23,(col+1)*18,row*23+18],row,textar={'bold':.2})
           for row in range(3) for col in range(10)]
    for i in (8,11):
        units[i].bold=True
        units[i].bold_confidence=.8
    assert wrapped_bold_candidates(units)=={9,10}
    units[10].textar={'bold':.01}
    assert wrapped_bold_candidates(units)==set()
    units[10].textar={'bold':.2}
    units[10].box[1]+=15
    units[10].box[3]+=15
    assert wrapped_bold_candidates(units)==set()
    units[10].box=[36,23,54,41]
    assert wrapped_bold_candidates(units)==set()


def test_validation_families_have_disjoint_content_and_fixed_split():
    from policy_validation_fixtures import TEXTS,MODES

    assert len(MODES)==8
    assert set(TEXTS)=={'development','holdout'}
    development={phrase for section in TEXTS['development'] for phrase in section}
    holdout={phrase for section in TEXTS['holdout'] for phrase in section}
    assert development.isdisjoint(holdout)


def test_html_table_bold_is_actually_rendered_and_not_just_star_syntax():
    from evaluate_api_dense import rendered_characters

    source='<table><tr><td>重点内容</td></tr></table>'
    units=[Unit(c,[i*12,0,i*12+11,18],0,bold=True,bold_confidence=.8) for i,c in enumerate('重点内容')]
    units[0].colored=True
    units[0].color='#0080ff'
    markdown,_=enrich_vl_markdown(source,units)
    assert '**' not in markdown
    assert '<strong>' in markdown
    chars,styles=rendered_characters(markdown)
    assert ''.join(chars)=='重点内容'
    assert all(s['bold'] for s in styles)
    _,incorrect=rendered_characters('<table><tr><td>**重点内容**</td></tr></table>')
    assert not any(s['bold'] for s in incorrect)
    _,ordinary=rendered_characters('普通文字\n\n**重点内容**')
    assert not any(s['bold'] for s in ordinary[:4])
    assert all(s['bold'] for s in ordinary[4:])
    punctuation_units=[Unit(c,[i*12,0,i*12+11,18],0,bold=i<3,bold_confidence=.8) for i,c in enumerate('重点。普通')]
    markup,_=enrich_vl_markdown('重点。普通',punctuation_units)
    chars,actual=rendered_characters(markup)
    assert ''.join(chars)=='重点。普通'
    assert [s['bold'] for s in actual]==[True,True,True,False,False]
