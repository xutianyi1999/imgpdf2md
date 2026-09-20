#!/usr/bin/env python3
"""Feasibility demo: AI Studio OCR plus local visual style recovery."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
FONTDNA_URL = (
    "https://huggingface.co/ClosedBrain/FontDNA-V2/resolve/main/"
    "glyphdna.int8.onnx"
)


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class AIStudioClient:
    def __init__(self, token: str, poll_interval: float = 3.0) -> None:
        self.headers = {"Authorization": f"bearer {token}"}
        self.poll_interval = poll_interval

    def run(self, file_path: Path, model: str, options: dict[str, Any]) -> bytes:
        with file_path.open("rb") as handle:
            response = requests.post(
                JOB_URL,
                headers=self.headers,
                data={"model": model, "optionalPayload": json.dumps(options)},
                files={"file": (file_path.name, handle)},
                timeout=120,
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"AI Studio submit failed: {payload.get('msg')}")
        job_id = payload["data"]["jobId"]

        deadline = time.monotonic() + 15 * 60
        while time.monotonic() < deadline:
            status_response = requests.get(
                f"{JOB_URL}/{job_id}", headers=self.headers, timeout=60
            )
            status_response.raise_for_status()
            status = status_response.json()["data"]
            state = status.get("state")
            if state == "done":
                result_response = requests.get(
                    status["resultUrl"]["jsonUrl"], timeout=180
                )
                result_response.raise_for_status()
                return result_response.content
            if state == "failed":
                raise RuntimeError(f"AI Studio job failed: {status.get('errorMsg')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"AI Studio job timed out: {job_id}")


class FontDNA:
    """Small adapter around the model-card ONNX interface."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))

    def predict(self, crop_bgr: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if height < 2 or width < 2:
            return {}
        scaled_width = max(8, min(320, round(width * 40 / height)))
        scaled_width = max(8, int(math.ceil(scaled_width / 8) * 8))
        resized = cv2.resize(gray, (scaled_width, 40), interpolation=cv2.INTER_AREA)
        array = resized.astype(np.float32) / 255.0
        array = (array - array.mean()) / (array.std() + 1e-4)
        inputs: dict[str, np.ndarray] = {
            "img": array[None, None, :, :],
            "cols": np.array([scaled_width // 8], dtype=np.int64),
        }
        inputs = {key: value for key, value in inputs.items() if key in self.input_names}
        outputs = self.session.run(None, inputs)
        styles = self._sigmoid(np.asarray(outputs[1]))[0]
        confidence = float(self._sigmoid(np.asarray(outputs[3])).reshape(-1)[0])
        return {
            "bold": float(styles[0]),
            "italic": float(styles[1]),
            "underline": float(styles[2]),
            "strikethrough": float(styles[3]),
            "confidence": confidence,
        }


def ensure_fontdna(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FontDNA-V2 to {model_path} ...", file=sys.stderr)
    with requests.get(FONTDNA_URL, stream=True, timeout=180) as response:
        response.raise_for_status()
        temporary = model_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        temporary.replace(model_path)
    return model_path


@dataclass
class Unit:
    text: str
    box: list[int]
    line_index: int
    color: str | None = None
    colored: bool = False
    color_confidence: float = 0.0
    underline: bool | None = None
    underline_confidence: float = 0.0
    bold: bool | None = None
    bold_confidence: float = 0.0
    fontdna: dict[str, float] | None = None
    textar: dict[str, float] | None = None
    stroke_width: float | None = None
    morphology_bold_score: float | None = None
    visual_font_height_px: int | None = None
    relative_size: str | None = None


def read_first_jsonl(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    raise ValueError(f"No JSON object in {path}")


def extract_ppocr(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    if not result.get("ocrResults"):
        raise ValueError("PP-OCR result has no ocrResults")
    pruned = result["ocrResults"][0]["prunedResult"]
    if not pruned.get("return_word_box"):
        raise ValueError(
            "PP-OCR result has no word boxes; call it with returnWordBox=true"
        )
    return pruned


def extract_vl_markdown(payload: dict[str, Any]) -> str:
    results = payload.get("result", {}).get("layoutParsingResults", [])
    return "\n\n".join(item.get("markdown", {}).get("text", "") for item in results)


def clamp_box(box: list[int], width: int, height: int, pad_x: int = 0, pad_y: int = 0) -> list[int]:
    x1, y1, x2, y2 = map(int, box)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    ]


def crop_box(image: np.ndarray, box: list[int], pad_ratio: float = 0.08) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(1, round((x2 - x1) * pad_ratio))
    pad_y = max(1, round((y2 - y1) * pad_ratio))
    x1, y1, x2, y2 = clamp_box(box, width, height, pad_x, pad_y)
    return image[y1:y2, x1:x2]


def estimate_color(image: np.ndarray, box: list[int]) -> tuple[str | None, bool, float]:
    crop = crop_box(image, box, 0.04)
    if crop.size == 0:
        return None, False, 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    delta = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    foreground = delta > max(24.0, float(np.percentile(delta, 65)) * 0.45)
    pixels = rgb[foreground]
    if len(pixels) < 5:
        return None, False, 0.0
    strengths = np.linalg.norm(pixels.astype(np.float32) - background, axis=1)
    # Dense small text has many light antialiasing pixels whose per-channel
    # rounding can look falsely tinted. Estimate from the strongest foreground
    # core instead of the median edge pixel.
    pixels = pixels[strengths >= np.percentile(strengths, 72)]
    color = np.median(pixels, axis=0).astype(int)
    red, green, blue = map(int, color)
    maximum, minimum = max(color), min(color)
    chroma = int(maximum - minimum)
    # A higher chroma floor avoids treating browser subpixel antialiasing on
    # dense gray/black text as an intentional font color. The target documents
    # use clearly accented link/warning colors; muted gray remains normal text.
    colored = bool(chroma >= 95 and maximum >= 75)
    confidence = min(1.0, max(0.0, (chroma - 15) / 80)) if colored else 0.9
    return f"#{red:02x}{green:02x}{blue:02x}", colored, confidence


def harmonize_line_colors(units: list[Unit]) -> None:
    """Merge anti-aliased estimates that represent the same source color."""
    for line_index in sorted({unit.line_index for unit in units}):
        colored = [
            unit for unit in units
            if unit.line_index == line_index and unit.colored and unit.color
        ]
        remaining = set(range(len(colored)))
        while remaining:
            component = {remaining.pop()}
            changed = True
            while changed:
                changed = False
                for candidate in list(remaining):
                    candidate_rgb = np.array([
                        int(colored[candidate].color[index:index + 2], 16)  # type: ignore[index]
                        for index in (1, 3, 5)
                    ])
                    if any(
                        np.linalg.norm(candidate_rgb - np.array([
                            int(colored[member].color[index:index + 2], 16)  # type: ignore[index]
                            for index in (1, 3, 5)
                        ])) <= 70
                        for member in component
                    ):
                        remaining.remove(candidate)
                        component.add(candidate)
                        changed = True
            rgbs = np.array([
                [int(colored[item].color[index:index + 2], 16) for index in (1, 3, 5)]  # type: ignore[index]
                for item in component
            ])
            red, green, blue = np.median(rgbs, axis=0).astype(int)
            canonical = f"#{red:02x}{green:02x}{blue:02x}"
            for item in component:
                colored[item].color = canonical

        line_units = [unit for unit in units if unit.line_index == line_index]
        for index, unit in enumerate(line_units):
            if not unit.colored or any(char.isalnum() for char in unit.text):
                continue
            neighbors = line_units[max(0, index - 1):index] + line_units[index + 1:index + 2]
            if not any(neighbor.colored for neighbor in neighbors):
                # Isolated punctuation is particularly vulnerable to RGB
                # subpixel antialiasing and should inherit no accent color.
                unit.colored = False


def foreground_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    # Absolute contrast supports both dark text on light paper and light text
    # on dark UI/PDF backgrounds.
    return (np.abs(gray.astype(np.float32) - background) > 28).astype(np.uint8)


def stroke_width(crop: np.ndarray) -> float | None:
    mask = foreground_mask(crop)
    if int(mask.sum()) < 8:
        return None
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local_max = distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-5
    samples = distance[(mask > 0) & local_max & (distance > 0)]
    if samples.size == 0:
        return None
    return float(2.0 * np.median(samples))


def morphology_bold_score(crop: np.ndarray, target_height: int = 40) -> float | None:
    """Return a scale-normalized visual stroke-mass score.

    Eroding the foreground once removes thin strokes faster than thick strokes.
    The surviving-area ratio is not an absolute definition of bold, but is a
    useful within-line comparison for words rendered by the same engine.
    """
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    width = max(2, round(crop.shape[1] * target_height / crop.shape[0]))
    normalized = cv2.resize(crop, (width, target_height), interpolation=cv2.INTER_CUBIC)
    mask = foreground_mask(normalized)
    area = int(mask.sum())
    if area < 8:
        return None
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return float(eroded.sum() / area)


def underline_segments(image: np.ndarray, line_box: list[int]) -> list[list[int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, line_box)
    line_height = max(1, y2 - y1)
    roi_box = clamp_box(
        [x1, int(y1 + 0.62 * line_height), x2, int(y2 + 0.38 * line_height)],
        width,
        height,
    )
    rx1, ry1, rx2, ry2 = roi_box
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    binary = (np.abs(gray.astype(np.float32) - background) > 28).astype(np.uint8)
    # Underlines are continuous pixel runs. Looking for actual row runs is less
    # prone to joining neighboring CJK bottom strokes than a large morphology
    # kernel. Close only one-pixel anti-aliasing gaps.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    # High-precision demo threshold. Short word/single-character underline is
    # deliberately left to the model/VL candidate path because CJK bottom
    # strokes can otherwise create many false positives.
    minimum_width = max(12, round(line_height * 1.20))
    row_runs: list[list[int]] = []
    for row_index, row in enumerate(binary):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            absolute_y = ry1 + row_index
            if end - start < minimum_width:
                continue
            if absolute_y < y1 + line_height * 0.62:
                continue
            row_runs.append([rx1 + int(start), absolute_y, rx1 + int(end), absolute_y + 1])

    merged: list[list[int]] = []
    for run in row_runs:
        match = next(
            (
                item
                for item in merged
                if run[1] <= item[3] + 1
                and max(run[2] - run[0], item[2] - item[0])
                <= 1.35 * min(run[2] - run[0], item[2] - item[0])
                and max(0, min(run[2], item[2]) - max(run[0], item[0]))
                >= 0.7 * min(run[2] - run[0], item[2] - item[0])
            ),
            None,
        )
        if match:
            match[0] = min(match[0], run[0])
            match[2] = max(match[2], run[2])
            match[3] = max(match[3], run[3])
        else:
            merged.append(run)

    return [
        segment for segment in merged
        if (
            1 <= segment[3] - segment[1] <= max(4, line_height * 0.18)
            and segment[1] >= y2 - max(5, line_height * 0.16)
        )
    ]


def overlap_ratio(box: list[int], segment: list[int]) -> float:
    overlap = max(0, min(box[2], segment[2]) - max(box[0], segment[0]))
    return overlap / max(1, box[2] - box[0])


def build_units(pruned: dict[str, Any]) -> tuple[list[Unit], list[list[int]]]:
    units: list[Unit] = []
    line_boxes = pruned["rec_boxes"]
    for line_index, (words, boxes) in enumerate(
        zip(pruned["text_word"], pruned["text_word_boxes"])
    ):
        for text, box in zip(words, boxes):
            units.append(Unit(str(text), list(map(int, box)), line_index))
    return units, [list(map(int, box)) for box in line_boxes]


def textar_span_candidates(
    units: list[Unit],
    low_probability: float = 0.25,
    strong_probability: float = 0.55,
    minimum_mean: float = 0.40,
) -> set[int]:
    """Select coherent bold spans while rejecting isolated TexTAR spikes."""
    selected: set[int] = set()

    def probability(index: int) -> float:
        return float((units[index].textar or {}).get("bold", -1.0))

    def adjacent(left: Unit, right: Unit) -> bool:
        if left.line_index != right.line_index:
            return False
        left_width = max(1, left.box[2] - left.box[0])
        right_width = max(1, right.box[2] - right.box[0])
        gap = right.box[0] - left.box[2]
        return -0.5 * min(left_width, right_width) <= gap <= 0.85 * max(
            left_width, right_width
        )

    def accept(segment: list[int]) -> None:
        if not segment:
            return
        scores = [probability(index) for index in segment]
        character_count = sum(
            sum(character.isalnum() for character in units[index].text)
            for index in segment
        )
        if (
            character_count >= 2
            and max(scores) >= strong_probability
            and float(np.mean(scores)) >= minimum_mean
        ):
            selected.update(segment)

    for line_index in sorted({unit.line_index for unit in units}):
        line_indices = [
            index for index, unit in enumerate(units) if unit.line_index == line_index
        ]
        segment: list[int] = []
        for index in line_indices:
            eligible = (
                any(character.isalnum() for character in units[index].text)
                and probability(index) >= low_probability
            )
            if eligible and (
                not segment or adjacent(units[segment[-1]], units[index])
            ):
                segment.append(index)
            else:
                accept(segment)
                segment = [index] if eligible else []
        accept(segment)
    return selected


def classify_units(
    image: np.ndarray,
    units: list[Unit],
    line_boxes: list[list[int]],
    fontdna: FontDNA | None,
    textar: Any | None = None,
    *,
    preserve_span_confidence: bool = True,
) -> None:
    segments_by_line = [underline_segments(image, box) for box in line_boxes]
    line_strokes: dict[int, list[float]] = {}
    heights = [unit.box[3] - unit.box[1] for unit in units if any(char.isalnum() for char in unit.text)]
    body_height = float(np.median(heights)) if heights else 1.0

    for unit in units:
        unit.visual_font_height_px = max(1, unit.box[3] - unit.box[1])
        size_ratio = unit.visual_font_height_px / max(1.0, body_height)
        unit.relative_size = "small" if size_ratio < 0.82 else "large" if size_ratio > 1.24 else "body"
        unit.color, unit.colored, unit.color_confidence = estimate_color(image, unit.box)
        crop = crop_box(image, unit.box, 0.12)
        unit.stroke_width = stroke_width(crop)
        if unit.stroke_width is not None:
            line_strokes.setdefault(unit.line_index, []).append(unit.stroke_width)

        line_height = max(1, line_boxes[unit.line_index][3] - line_boxes[unit.line_index][1])
        plausible_segments = [
            segment for segment in segments_by_line[unit.line_index]
            if segment[2] - segment[0] >= 5.0 * line_height
        ]
        overlaps = [overlap_ratio(unit.box, segment) for segment in plausible_segments]
        best_overlap = max(overlaps, default=0.0)
        unit.underline = best_overlap >= 0.55
        unit.underline_confidence = min(1.0, best_overlap) if unit.underline else 0.9

    harmonize_line_colors(units)

    if textar is not None:
        try:
            for unit, prediction in zip(units, textar.predict(image, units)):
                unit.textar = prediction
        except Exception as exc:
            print(f"TexTAR inference skipped: {exc}", file=sys.stderr)
    textar_bold_indices = textar_span_candidates(units)

    # FontDNA is explicitly a word-crop model. PP-OCR returns Chinese as single
    # characters, so use jieba only to form visual word crops; OCR text remains
    # untouched. The same groups are used by the morphology comparator.
    import jieba

    morphology_baselines: dict[int, float] = {}
    for line_index in range(len(line_boxes)):
            line_units = [unit for unit in units if unit.line_index == line_index]
            line_text = "".join(unit.text for unit in line_units)
            token_ranges: list[tuple[int, int, str]] = []
            cursor = 0
            for token in jieba.lcut(line_text, cut_all=False):
                if not token:
                    continue
                token_ranges.append((cursor, cursor + len(token), token))
                cursor += len(token)
            offsets = []
            cursor = 0
            for unit in line_units:
                offsets.append((cursor, cursor + len(unit.text), unit))
                cursor += len(unit.text)
            group_scores: list[float] = []
            for start, end, token in token_ranges:
                members = [unit for left, right, unit in offsets if right > start and left < end]
                if not members or token.isspace():
                    continue
                group_box = [
                    min(unit.box[0] for unit in members),
                    min(unit.box[1] for unit in members),
                    max(unit.box[2] for unit in members),
                    max(unit.box[3] for unit in members),
                ]
                group_crop = crop_box(image, group_box, 0.10)
                score = morphology_bold_score(group_crop)
                if score is not None and any(char.isalnum() for char in token):
                    group_scores.append(score)
                for unit in members:
                    unit.morphology_bold_score = score
                if fontdna is not None:
                    try:
                        prediction = fontdna.predict(group_crop)
                        prediction["group_text"] = token  # type: ignore[assignment]
                        for unit in members:
                            unit.fontdna = dict(prediction)
                    except Exception as exc:  # Keep visual algorithms usable.
                        for unit in members:
                            unit.fontdna = {"error": str(exc)}  # type: ignore[dict-item]
            if len(group_scores) >= 3:
                morphology_baselines[line_index] = float(np.percentile(group_scores, 25))

    for unit_index, unit in enumerate(units):
        model_bold = None
        model_conf = 0.0
        if unit.fontdna and "bold" in unit.fontdna:
            probability = float(unit.fontdna["bold"])
            model_quality = float(unit.fontdna.get("confidence", 1.0))
            group_text = str(unit.fontdna.get("group_text", ""))
            # Single Han glyphs were unstable in the real screenshot. Keep a
            # conservative high-precision threshold for this feasibility demo.
            if len(group_text) >= 2 and probability >= 0.90:
                model_bold = True
            elif len(group_text) >= 2 and probability <= 0.20:
                model_bold = False
            model_conf = abs(probability - 0.5) * 2 * model_quality

        strokes = line_strokes.get(unit.line_index, [])
        heuristic_bold = None
        heuristic_conf = 0.0
        if unit.stroke_width is not None and len(strokes) >= 4:
            baseline = float(np.percentile(strokes, 35))
            ratio = unit.stroke_width / max(0.1, baseline)
            heuristic_bold = ratio >= 1.20
            heuristic_conf = min(0.85, abs(ratio - 1.1))

        morphology_bold = None
        morphology_conf = 0.0
        baseline = morphology_baselines.get(unit.line_index)
        if unit.morphology_bold_score is not None and baseline is not None:
            delta = unit.morphology_bold_score - baseline
            fontdna_probability = float((unit.fontdna or {}).get("bold", -1.0))
            textar_probability = float((unit.textar or {}).get("bold", -1.0))
            joint_support = fontdna_probability >= 0.35 or textar_probability >= 0.25
            if unit.morphology_bold_score >= 0.30 and delta >= 0.24:
                morphology_bold = True
            elif (
                unit.morphology_bold_score >= 0.60
                and delta >= 0.14
                and joint_support
            ):
                morphology_bold = True
            elif delta <= 0.08:
                morphology_bold = False
            morphology_conf = min(0.95, max(0.0, (delta - 0.12) / 0.18))
            if morphology_bold is True:
                morphology_conf = max(morphology_conf, 0.55)

        textar_bold = None
        textar_conf = 0.0
        if unit.textar and "bold" in unit.textar:
            probability = float(unit.textar["bold"])
            fontdna_probability = float((unit.fontdna or {}).get("bold", -1.0))
            # Calibrated for high precision on the source-labelled dense suite.
            # A single character needs agreement from the independent word
            # crop model; coherent spans can safely recover lower-score gaps.
            if unit_index in textar_bold_indices or (
                probability >= 0.60 and fontdna_probability >= 0.40
            ):
                textar_bold = True
            textar_conf = min(1.0, max(0.0, (probability - 0.40) / 0.35))
            if preserve_span_confidence and unit_index in textar_bold_indices:
                # The span has already passed contextual acceptance. Using
                # only the individual score here silently undoes gap filling
                # when style_key applies its rendering confidence threshold.
                textar_conf = max(textar_conf, 0.25)

        # Union of independent high-precision signals. TexTAR supplies page
        # context, FontDNA covers word crops, and relative morphology recovers
        # inline bold where a normal-weight baseline exists on the same line.
        if model_bold is True or morphology_bold is True or textar_bold is True:
            unit.bold = True
            unit.bold_confidence = max(
                model_conf if model_bold else 0.0,
                morphology_conf if morphology_bold else 0.0,
                textar_conf if textar_bold else 0.0,
            )
        elif model_bold is False and morphology_bold is False:
            unit.bold = False
            unit.bold_confidence = max(model_conf, morphology_conf)
        else:
            unit.bold = None
            unit.bold_confidence = max(model_conf, morphology_conf, heuristic_conf)


def quantize_color(color: str) -> str:
    values = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    values = [min(255, round(value / 16) * 16) for value in values]
    return "#" + "".join(f"{value:02x}" for value in values)


def style_key(unit: Unit) -> tuple[bool, bool, str | None]:
    bold = bool(unit.bold and unit.bold_confidence >= 0.20)
    underline = bool(unit.underline)
    color = quantize_color(unit.color) if unit.colored and unit.color else None
    return bold, underline, color


def render_segment(text: str, key: tuple[bool, bool, str | None]) -> str:
    bold, underline, color = key
    rendered = text
    if color:
        rendered = f'<span style="color:{color}">{rendered}</span>'
    if underline:
        rendered = f"<u>{rendered}</u>"
    if bold:
        rendered = f"**{rendered}**"
    return rendered


def make_spans(units: list[Unit]) -> tuple[list[dict[str, Any]], str]:
    if not units:
        return [], ""
    spans: list[dict[str, Any]] = []
    markdown_parts: list[str] = []
    offset = 0
    start = 0
    current_key = style_key(units[0])
    text_parts: list[str] = []
    span_units: list[Unit] = []

    def append_span(text: str, end: int) -> None:
        states = [item.bold for item in span_units]
        bold_state: bool | None = True if current_key[0] else False if states and all(state is False for state in states) else None
        spans.append({
            "start": start, "end": end, "text": text,
            "bold": bold_state, "render_bold": current_key[0],
            "underline": current_key[1], "color": current_key[2],
        })
        markdown_parts.append(render_segment(text, current_key))

    for index, unit in enumerate(units):
        key = style_key(unit)
        if key != current_key and text_parts:
            text = "".join(text_parts)
            append_span(text, offset)
            start = offset
            text_parts = []
            span_units = []
            current_key = key
        text_parts.append(unit.text)
        span_units.append(unit)
        offset += len(unit.text)
        if index == len(units) - 1:
            text = "".join(text_parts)
            append_span(text, offset)
    return spans, "".join(markdown_parts)


def create_result(units: list[Unit], line_count: int) -> tuple[dict[str, Any], str]:
    lines = []
    markdown_lines = []
    for line_index in range(line_count):
        line_units = [unit for unit in units if unit.line_index == line_index]
        spans, markdown = make_spans(line_units)
        lines.append({
            "line_index": line_index,
            "text": "".join(unit.text for unit in line_units),
            "spans": spans,
            "units": [asdict(unit) for unit in line_units],
        })
        markdown_lines.append(markdown)
    return {"offset_unit": "unicode_code_point", "lines": lines}, "\n\n".join(markdown_lines)


def _normalized_chars(text: str, owners: list[Any]) -> tuple[list[str], list[Any]]:
    chars: list[str] = []
    expanded_owners: list[Any] = []
    for char, owner in zip(text, owners):
        for normalized in unicodedata.normalize("NFKC", char):
            if not normalized.isspace():
                chars.append(normalized)
                expanded_owners.append(owner)
    return chars, expanded_owners


def _markdown_visible(markdown: str) -> tuple[list[str], list[int]]:
    """Return visible characters and source indices for common VL Markdown."""
    visible: list[str] = []
    indices: list[int] = []
    in_tag = False
    link_target_depth = 0
    line_start = True
    index = 0
    while index < len(markdown):
        char = markdown[index]
        if char == "\n":
            line_start = True
            index += 1
            continue
        if line_start and char in " \t":
            index += 1
            continue
        if line_start and char == "#":
            while index < len(markdown) and markdown[index] == "#":
                index += 1
            if index < len(markdown) and markdown[index] == " ":
                index += 1
            line_start = False
            continue
        line_start = False
        if char == "<":
            in_tag = True
        if in_tag:
            if char == ">":
                in_tag = False
            index += 1
            continue
        if markdown.startswith("](", index):
            link_target_depth = 1
            index += 2
            continue
        if link_target_depth:
            if char == "(":
                link_target_depth += 1
            elif char == ")":
                link_target_depth -= 1
            index += 1
            continue
        if markdown.startswith("**", index) or markdown.startswith("__", index):
            index += 2
            continue
        if char in "[]`":
            index += 1
            continue
        normalized = unicodedata.normalize("NFKC", char)
        for item in normalized:
            if not item.isspace():
                visible.append(item)
                indices.append(index)
        index += 1
    return visible, indices


def enrich_vl_markdown(markdown: str, units: list[Unit]) -> tuple[str, float]:
    """Align PP-OCR styles onto VL Markdown while preserving VL structure."""
    # VL commonly expresses a visual underline as inline LaTeX. HTML <u> is
    # composable with color spans and renders in ordinary Markdown previews.
    markdown = re.sub(
        r"\$\s*\\underline\{\\text\{(.*?)\}\}\s*\$",
        r"<u>\1</u>",
        markdown,
        flags=re.DOTALL,
    )
    vl_chars, vl_indices = _markdown_visible(markdown)
    pp_text = "".join(unit.text for unit in units)
    pp_owners: list[Unit] = []
    for unit in units:
        pp_owners.extend([unit] * len(unit.text))
    pp_chars, pp_owners = _normalized_chars(pp_text, pp_owners)
    matcher = difflib.SequenceMatcher(None, vl_chars, pp_chars, autojunk=False)
    mapped: dict[int, tuple[bool, bool, str | None]] = {}
    semantic_bold: set[int] = set()
    semantic_underline: set[int] = set()
    for match in re.finditer(r"(?m)^#{1,6} +([^\n]+)", markdown):
        semantic_bold.update(range(match.start(1), match.end(1)))
    for match in re.finditer(r"\*\*(.+?)\*\*", markdown, re.DOTALL):
        semantic_bold.update(range(match.start(1), match.end(1)))
    for match in re.finditer(r"<u>(.*?)</u>", markdown, re.DOTALL):
        semantic_underline.update(range(match.start(1), match.end(1)))
    matched = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            source_index = vl_indices[block.a + offset]
            bold, underline, color = style_key(pp_owners[block.b + offset])
            mapped[source_index] = (
                bold and source_index not in semantic_bold,
                underline and source_index not in semantic_underline,
                color,
            )
            matched += 1

    for index, char in enumerate(markdown):
        if not char.isspace() or char == "\n":
            continue
        left = mapped.get(index - 1)
        right = mapped.get(index + 1)
        if left is not None and left == right:
            mapped[index] = left

    output: list[str] = []
    active: tuple[bool, bool, str | None] | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal active, buffer
        if buffer:
            output.append(render_segment("".join(buffer), active or (False, False, None)))
        active, buffer = None, []

    for index, char in enumerate(markdown):
        key = mapped.get(index)
        if char == "\n" or key is None:
            flush()
            output.append(char)
        elif active == key:
            buffer.append(char)
        else:
            flush()
            active = key
            buffer.append(char)
    flush()
    return "".join(output), matched / max(1, len(vl_chars))


def draw_overlay(image: np.ndarray, units: list[Unit]) -> np.ndarray:
    overlay = image.copy()
    for unit in units:
        x1, y1, x2, y2 = unit.box
        key = style_key(unit)
        if key[1]:
            color = (0, 170, 0)
        elif key[0]:
            color = (0, 0, 230)
        elif key[2]:
            color = (230, 120, 0)
        else:
            color = (160, 160, 160)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        labels = []
        if key[0]: labels.append("B")
        if key[1]: labels.append("U")
        if key[2]: labels.append(key[2])
        if labels:
            cv2.putText(overlay, "/".join(labels), (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return overlay


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    import pymupdf

    page_dir = output_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    scale = dpi / 72.0
    paths = []
    for index, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        path = page_dir / f"page-{index + 1:04d}.png"
        pixmap.save(path)
        paths.append(path)
    return paths


def get_or_call(
    client: AIStudioClient | None,
    image_path: Path,
    output_path: Path,
    supplied: Path | None,
    model: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    if supplied:
        return read_first_jsonl(supplied)
    if output_path.exists():
        return read_first_jsonl(output_path)
    if client is None:
        raise ValueError("TOKEN is required when API JSONL is not supplied or cached")
    content = client.run(image_path, model, options)
    output_path.write_bytes(content)
    return read_first_jsonl(output_path)


def process_page(
    image_path: Path,
    output_dir: Path,
    client: AIStudioClient | None,
    fontdna: FontDNA | None,
    textar: Any | None = None,
    ppocr_jsonl: Path | None = None,
    vl_jsonl: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ppocr = get_or_call(
        client,
        image_path,
        output_dir / "ppocr.jsonl",
        ppocr_jsonl,
        "PP-OCRv6",
        {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
            "returnWordBox": True,
        },
    )
    vl = get_or_call(
        client,
        image_path,
        output_dir / "vl.jsonl",
        vl_jsonl,
        "PaddleOCR-VL-1.6",
        {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        },
    )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    pruned = extract_ppocr(ppocr)
    source_width = int(ppocr.get("result", {}).get("dataInfo", {}).get("width", image.shape[1]))
    source_height = int(ppocr.get("result", {}).get("dataInfo", {}).get("height", image.shape[0]))
    if (source_width, source_height) != (image.shape[1], image.shape[0]):
        scale_x = image.shape[1] / source_width
        scale_y = image.shape[0] / source_height
        for key in ("rec_boxes", "text_word_boxes"):
            values = pruned.get(key, [])
            if key == "rec_boxes":
                for box in values:
                    box[:] = [round(box[0] * scale_x), round(box[1] * scale_y), round(box[2] * scale_x), round(box[3] * scale_y)]
            else:
                for line in values:
                    for box in line:
                        box[:] = [round(box[0] * scale_x), round(box[1] * scale_y), round(box[2] * scale_x), round(box[3] * scale_y)]

    units, line_boxes = build_units(pruned)
    classify_units(image, units, line_boxes, fontdna, textar)
    result, styled_ocr_markdown = create_result(units, len(line_boxes))
    vl_markdown = extract_vl_markdown(vl)
    styled_markdown, alignment_coverage = enrich_vl_markdown(vl_markdown, units)
    result["source_image"] = str(image_path)
    result["vl_alignment_coverage"] = alignment_coverage
    result["note"] = "Feasibility output; styles require validation on representative data."
    (output_dir / "styles.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "styled.md").write_text(styled_markdown, encoding="utf-8")
    (output_dir / "styled_ocr.md").write_text(styled_ocr_markdown, encoding="utf-8")
    (output_dir / "vl.md").write_text(vl_markdown, encoding="utf-8")
    cv2.imwrite(str(output_dir / "overlay.jpg"), draw_overlay(image, units))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PNG/JPEG or image PDF")
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--no-fontdna", action="store_true")
    parser.add_argument("--fontdna-model", type=Path, default=Path(".cache/fontdna/glyphdna.int8.onnx"))
    parser.add_argument("--textar", action="store_true", help="Enable the optional context-aware TexTAR backend")
    parser.add_argument("--textar-model", type=Path, default=Path(".cache/textar/TexTAR-trained.pt"))
    parser.add_argument("--ppocr-jsonl", type=Path)
    parser.add_argument("--vl-jsonl", type=Path)
    args = parser.parse_args()

    env = load_dotenv(args.env)
    token = env.get("TOKEN") or os.environ.get("PADDLEOCR_ACCESS_TOKEN")
    client = AIStudioClient(token) if token else None
    model = None if args.no_fontdna else FontDNA(ensure_fontdna(args.fontdna_model))
    textar = None
    if args.textar:
        try:
            from textar_backend import TexTARBackend, ensure_textar

            textar = TexTARBackend(ensure_textar(args.textar_model))
        except ImportError as exc:
            parser.error(f"TexTAR dependencies are missing; run with the textar dependency group: {exc}")

    stem_dir = args.output / args.input.stem
    if args.input.suffix.lower() == ".pdf":
        pages = render_pdf(args.input, stem_dir, args.dpi)
        if args.ppocr_jsonl or args.vl_jsonl:
            parser.error("JSONL overrides are only supported for a single image")
    else:
        pages = [args.input]

    for index, page_path in enumerate(pages, start=1):
        page_output = stem_dir / f"page-{index:04d}"
        print(f"Processing {page_path} -> {page_output}", file=sys.stderr)
        process_page(
            page_path,
            page_output,
            client,
            model,
            textar,
            args.ppocr_jsonl if len(pages) == 1 else None,
            args.vl_jsonl if len(pages) == 1 else None,
        )
    print(stem_dir)


if __name__ == "__main__":
    main()
