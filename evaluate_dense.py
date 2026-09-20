#!/usr/bin/env python3
"""Evaluate style algorithms on dense-text screenshot fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from dense_fixtures import generate_dense
from eval_utils import assign_lines, binary_metrics
from style_demo import FontDNA, classify_units, ensure_fontdna, style_key
from textar_backend import TexTARBackend, ensure_textar


def main() -> None:
    root = Path("testdata/dense")
    manifest = json.loads(generate_dense(root).read_text(encoding="utf-8"))
    model = FontDNA(ensure_fontdna(Path(".cache/fontdna/glyphdna.int8.onnx")))
    textar = TexTARBackend(ensure_textar(Path(".cache/textar/TexTAR-trained.pt")))
    overall = {name: [] for name in ("bold", "underline", "colored")}
    baseline_overall = {name: [] for name in ("bold", "underline", "colored")}
    cases = []
    for case in manifest["cases"]:
        image = cv2.imread(str(root / case["image"]))
        baseline_units, baseline_lines = assign_lines(case["units"])
        classify_units(image, baseline_units, baseline_lines, model)
        units, lines = assign_lines(case["units"])
        classify_units(image, units, lines, model, textar)
        pairs = {name: [] for name in overall}
        for unit in units:
            truth = unit._truth  # type: ignore[attr-defined]
            bold, underline, color = style_key(unit)
            prediction = {"bold": bold, "underline": underline, "colored": color is not None}
            for name in pairs:
                pair = (prediction[name], bool(truth[name]))
                pairs[name].append(pair); overall[name].append(pair)
        for unit in baseline_units:
            truth = unit._truth  # type: ignore[attr-defined]
            bold, underline, color = style_key(unit)
            prediction = {"bold": bold, "underline": underline, "colored": color is not None}
            for name in baseline_overall:
                baseline_overall[name].append((prediction[name], bool(truth[name])))
        cases.append({"name":case["name"],"characters":len(units),"metrics":{name:binary_metrics(value) for name,value in pairs.items()}})
    report={
        "metrics": {name: binary_metrics(value) for name, value in overall.items()},
        "baseline_without_textar": {
            name: binary_metrics(value) for name, value in baseline_overall.items()
        },
        "cases": cases,
    }
    output=root/"evaluation.json";output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2));print(output)


if __name__ == "__main__": main()
