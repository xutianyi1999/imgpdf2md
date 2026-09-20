#!/usr/bin/env python3
"""Evaluate cached AI Studio + local-style output on dense target documents."""

from __future__ import annotations

import difflib
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

from eval_utils import binary_metrics


def rendered_characters(markdown: str) -> tuple[list[str], list[dict[str, bool]]]:
    chars: list[str] = []
    styles: list[dict[str, bool]] = []
    bold = False
    underline_depth = color_depth = 0
    heading = True
    line_start = True
    link_depth = 0
    index = 0
    while index < len(markdown):
        if markdown[index] == "\n":
            line_start, heading = True, False
            index += 1
            continue
        if line_start and markdown[index] == "#":
            heading = True
            while index < len(markdown) and markdown[index] == "#": index += 1
            while index < len(markdown) and markdown[index] == " ": index += 1
            line_start = False
            continue
        line_start = False
        if markdown.startswith("**", index):
            bold = not bold; index += 2; continue
        if markdown.startswith("<u>", index): underline_depth += 1; index += 3; continue
        if markdown.startswith("</u>", index): underline_depth = max(0, underline_depth - 1); index += 4; continue
        if markdown.startswith("<span", index):
            end = markdown.find(">", index)
            if end >= 0:
                color_depth += int("color:" in markdown[index:end]); index = end + 1; continue
        if markdown.startswith("</span>", index): color_depth = max(0, color_depth - 1); index += 7; continue
        if markdown[index] == "<":
            end = markdown.find(">", index)
            if end >= 0: index = end + 1; continue
        if markdown.startswith("](", index): link_depth = 1; index += 2; continue
        if link_depth:
            if markdown[index] == "(": link_depth += 1
            elif markdown[index] == ")": link_depth -= 1
            index += 1; continue
        char = markdown[index]
        if char in "[]`" or char.isspace(): index += 1; continue
        for normalized in unicodedata.normalize("NFKC", char):
            chars.append(normalized)
            styles.append({"bold": bold or heading, "underline": underline_depth > 0, "colored": color_depth > 0})
        index += 1
    return chars, styles


def intersection(a: list[int], b: list[int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def match_unit(truth: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    x1, y1, x2, y2 = truth["box"]
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    candidates = [unit for unit in predictions if unit["box"][0] <= center[0] <= unit["box"][2] and unit["box"][1] <= center[1] <= unit["box"][3]]
    if candidates:
        return min(candidates, key=lambda unit: (unit["box"][2] - unit["box"][0]) * (unit["box"][3] - unit["box"][1]))
    candidate = max(predictions, key=lambda unit: intersection(truth["box"], unit["box"]), default=None)
    return candidate if candidate and intersection(truth["box"], candidate["box"]) > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('demo_output'))
    args = parser.parse_args()
    truth = json.loads(Path("testdata/dense/ground_truth.json").read_text(encoding="utf-8"))
    reports = []
    for case in truth["cases"]:
        styles_path = args.output_root / case["name"] / "page-0001" / "styles.json"
        if not styles_path.exists():
            continue
        result = json.loads(styles_path.read_text(encoding="utf-8"))
        predictions = [unit for line in result["lines"] for unit in line["units"]]
        pairs = {name: [] for name in ("bold", "underline", "colored")}
        mapped = 0
        for expected in case["units"]:
            predicted = match_unit(expected, predictions)
            mapped += predicted is not None
            values = {
                "bold": bool(predicted and predicted["bold"] and predicted["bold_confidence"] >= 0.2),
                "underline": bool(predicted and predicted["underline"]),
                "colored": bool(predicted and predicted["colored"]),
            }
            for name in pairs:
                pairs[name].append((values[name], bool(expected[name])))
        truth_text = "".join(unit["text"] for unit in case["units"])
        ocr_text = "".join(unit["text"] for unit in predictions if not unit["text"].isspace())
        styled_markdown = (styles_path.parent / "styled.md").read_text(encoding="utf-8")
        rendered_chars, rendered_styles = rendered_characters(styled_markdown)
        final_pairs = {name: [] for name in pairs}
        truth_chars = list(truth_text)
        final_by_truth: dict[int, dict[str, bool]] = {}
        matcher = difflib.SequenceMatcher(None, truth_chars, rendered_chars, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                final_by_truth[block.a + offset] = rendered_styles[block.b + offset]
        for index, expected in enumerate(case["units"]):
            predicted = final_by_truth.get(index, {"bold": False, "underline": False, "colored": False})
            for name in final_pairs:
                final_pairs[name].append((predicted[name], bool(expected[name])))

        reports.append({
            "name": case["name"], "characters": len(case["units"]),
            "coordinate_coverage": mapped / len(case["units"]),
            "text_sequence_similarity": difflib.SequenceMatcher(None, truth_text, ocr_text, autojunk=False).ratio(),
            "vl_alignment_coverage": result.get("vl_alignment_coverage"),
            "metrics": {name: binary_metrics(value) for name, value in pairs.items()},
            "final_markdown_metrics": {name: binary_metrics(value) for name, value in final_pairs.items()},
        })
    output = args.output_root / 'dense_api_evaluation.json'
    output.write_text(json.dumps({"cases": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": reports}, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
