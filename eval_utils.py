"""Shared character-level evaluation helpers."""

from __future__ import annotations

from typing import Any

from style_demo import Unit


def binary_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, float | int]:
    tp = sum(predicted and expected for predicted, expected in pairs)
    fp = sum(predicted and not expected for predicted, expected in pairs)
    fn = sum(not predicted and expected for predicted, expected in pairs)
    tn = sum(not predicted and not expected for predicted, expected in pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def assign_lines(truth_units: list[dict[str, Any]]) -> tuple[list[Unit], list[list[int]]]:
    clusters: list[dict[str, Any]] = []
    for truth in sorted(truth_units, key=lambda item: (item["box"][1], item["box"][0])):
        x1, y1, x2, y2 = truth["box"]
        best = None
        best_overlap = 0.0
        for cluster in clusters:
            overlap = max(0, min(y2, cluster["y2"]) - max(y1, cluster["y1"]))
            ratio = overlap / max(1, min(y2 - y1, cluster["y2"] - cluster["y1"]))
            if ratio > best_overlap:
                best, best_overlap = cluster, ratio
        if best is None or best_overlap < 0.55:
            clusters.append({"y1": y1, "y2": y2, "items": [truth]})
        else:
            best["items"].append(truth)
            best["y1"] = min(best["y1"], y1)
            best["y2"] = max(best["y2"], y2)
    clusters.sort(key=lambda cluster: cluster["y1"])
    units: list[Unit] = []
    boxes: list[list[int]] = []
    for line_index, cluster in enumerate(clusters):
        items = sorted(cluster["items"], key=lambda item: item["box"][0])
        boxes.append([
            min(item["box"][0] for item in items), min(item["box"][1] for item in items),
            max(item["box"][2] for item in items), max(item["box"][3] for item in items),
        ])
        for truth in items:
            unit = Unit(truth["text"], truth["box"], line_index)
            unit._truth = truth  # type: ignore[attr-defined]
            units.append(unit)
    return units, boxes
