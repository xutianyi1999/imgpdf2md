#!/usr/bin/env python3
"""Evaluate local style recovery across the parameterized style matrix."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from eval_utils import assign_lines, binary_metrics
from style_demo import FontDNA, classify_units, ensure_fontdna, style_key
from style_matrix_fixtures import generate_style_matrix
from textar_backend import TexTARBackend, ensure_textar


def main() -> None:
    root = Path("testdata/style_matrix")
    manifest = json.loads(generate_style_matrix(root).read_text(encoding="utf-8"))
    fontdna = FontDNA(ensure_fontdna(Path(".cache/fontdna/glyphdna.int8.onnx")))
    textar = TexTARBackend(ensure_textar(Path(".cache/textar/TexTAR-trained.pt")))
    aggregate = {name: [] for name in ("bold", "underline", "colored")}
    groups: dict[str, dict[str, list[tuple[bool, bool]]]] = {}
    cases = []
    for case in manifest["cases"]:
        image = cv2.imread(str(root / case["image"]))
        units, lines = assign_lines(case["units"])
        classify_units(image, units, lines, fontdna, textar)
        pairs = {name: [] for name in aggregate}
        for unit in units:
            truth = unit._truth  # type: ignore[attr-defined]
            bold, underline, color = style_key(unit)
            prediction = {"bold": bold, "underline": underline, "colored": color is not None}
            for name in pairs:
                pair = prediction[name], bool(truth[name])
                pairs[name].append(pair)
                aggregate[name].append(pair)
                for group in (
                    f"font={case['font_family']}",
                    f"size={case['font_size']}",
                    f"weight={case['bold_weight']}",
                    f"theme={case['theme']}",
                ):
                    groups.setdefault(
                        group, {style: [] for style in aggregate}
                    )[name].append(pair)
        cases.append({
            "name": case["name"],
            "characters": len(units),
            "metrics": {name: binary_metrics(value) for name, value in pairs.items()},
        })
    report = {
        "metrics": {name: binary_metrics(value) for name, value in aggregate.items()},
        "group_metrics": {
            group: {name: binary_metrics(value) for name, value in styles.items()}
            for group, styles in sorted(groups.items())
        },
        "cases": cases,
    }
    output = root / "evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
