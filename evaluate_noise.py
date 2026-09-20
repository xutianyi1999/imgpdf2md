#!/usr/bin/env python3
"""Evaluate local style robustness across screenshot degradation variants."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from eval_utils import assign_lines, binary_metrics
from noisy_fixtures import generate_noisy
from style_demo import FontDNA, classify_units, ensure_fontdna, style_key


def main() -> None:
    fixture_dir = Path("testdata/noisy")
    manifest = json.loads(generate_noisy(fixture_dir).read_text(encoding="utf-8"))
    model = FontDNA(ensure_fontdna(Path(".cache/fontdna/glyphdna.int8.onnx")))
    aggregate = {
        variant: {name: [] for name in ("bold", "underline", "colored")}
        for variant in manifest["variants"]
    }
    cases = []
    for case in manifest["cases"]:
        image = cv2.imread(str(fixture_dir / case["image"]))
        units, line_boxes = assign_lines(case["units"])
        classify_units(image, units, line_boxes, model)
        pairs = {name: [] for name in ("bold", "underline", "colored")}
        for unit in units:
            truth = unit._truth  # type: ignore[attr-defined]
            bold, underline, color = style_key(unit)
            predicted = {"bold": bold, "underline": underline, "colored": color is not None}
            for name in pairs:
                pair = (predicted[name], bool(truth[name]))
                pairs[name].append(pair)
                aggregate[case["variant"]][name].append(pair)
        cases.append({
            "page": case["page"], "document": case["document"], "variant": case["variant"],
            "metrics": {name: binary_metrics(value) for name, value in pairs.items()},
        })
    report = {
        "variants": {
            variant: {name: binary_metrics(value) for name, value in styles.items()}
            for variant, styles in aggregate.items()
        },
        "cases": cases,
    }
    output = fixture_dir / "evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["variants"], ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
