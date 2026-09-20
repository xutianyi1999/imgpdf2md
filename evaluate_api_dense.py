#!/usr/bin/env python3
"""Evaluate cached AI Studio + local-style output on dense target documents."""

from __future__ import annotations

import difflib
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any
from html.parser import HTMLParser
from markdown_it import MarkdownIt

from eval_utils import binary_metrics


def rendered_characters(markdown: str) -> tuple[list[str], list[dict[str, bool]]]:
    """Measure rendered HTML, not a guessed interpretation of star delimiters."""
    chars: list[str] = []
    styles: list[dict[str, bool]] = []

    class VisibleHTML(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack=[]

        def handle_starttag(self,tag,attrs):
            if tag in {'br','img','hr','input','meta','link','wbr'}:
                return
            css=dict(attrs).get('style','') or ''
            state={'bold':tag in {'b','strong','th','h1','h2','h3','h4','h5','h6'},
                   'underline':tag=='u','colored':'color:' in css.replace(' ','')}
            self.stack.append((tag,state))

        def handle_endtag(self,tag):
            for i in range(len(self.stack)-1,-1,-1):
                if self.stack[i][0]==tag:
                    del self.stack[i:]
                    break

        def handle_data(self,data):
            if any(tag in {'script','style'} for tag,_ in self.stack):
                return
            state={key:any(value[key] for _,value in self.stack) for key in ('bold','underline','colored')}
            for char in unicodedata.normalize('NFKC',data):
                if not char.isspace():
                    chars.append(char)
                    styles.append(dict(state))

    reader=VisibleHTML()
    reader.feed(MarkdownIt('commonmark',{'html':True}).enable('table').render(markdown))
    reader.close()
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
    parser.add_argument('--truth', type=Path, default=Path('testdata/dense/ground_truth.json'))
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
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
        truth_chars = []
        truth_owners = []
        for index, expected in enumerate(case['units']):
            normalized = list(unicodedata.normalize('NFKC', expected['text']))
            truth_chars.extend(normalized)
            truth_owners.extend([index] * len(normalized))
        final_by_truth: dict[int, list[dict[str, bool]]] = {}
        matcher = difflib.SequenceMatcher(None, truth_chars, rendered_chars, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                owner = truth_owners[block.a + offset]
                final_by_truth.setdefault(owner, []).append(rendered_styles[block.b + offset])
        for index, expected in enumerate(case["units"]):
            matches = final_by_truth.get(index, [])
            required = len(unicodedata.normalize('NFKC', expected['text']))
            predicted = {name: len(matches) == required and bool(matches) and all(m[name] for m in matches) for name in final_pairs}
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
