#!/usr/bin/env python3
"""Derive labeled screenshot degradations from dense target documents."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from dense_fixtures import generate_dense


Transform = Callable[[np.ndarray, list[dict[str, Any]]], tuple[np.ndarray, list[dict[str, Any]]]]


def scale_boxes(units: list[dict[str, Any]], sx: float, sy: float) -> list[dict[str, Any]]:
    result = deepcopy(units)
    for unit in result:
        x1, y1, x2, y2 = unit["box"]
        unit["box"] = [round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)]
    return result


def clean(image: np.ndarray, units: list[dict[str, Any]]):
    return image.copy(), deepcopy(units)


def scaled_screenshot(image: np.ndarray, units: list[dict[str, Any]]):
    scale = 0.67
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale_boxes(units, scale, scale)


def mildly_scaled_screenshot(image: np.ndarray, units: list[dict[str, Any]]):
    scale = 0.85
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale_boxes(units, scale, scale)


def jpeg_screenshot(image: np.ndarray, units: list[dict[str, Any]]):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), deepcopy(units)


def messenger_jpeg(image: np.ndarray, units: list[dict[str, Any]]):
    # Chat applications often resize once and re-encode aggressively.
    small = cv2.resize(image, None, fx=0.78, fy=0.78, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 42])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded, scale_boxes(units, 0.78, 0.78)


def blur_and_noise(image: np.ndarray, units: list[dict[str, Any]]):
    rng = np.random.default_rng(20260920)
    blurred = cv2.GaussianBlur(image, (3, 3), 0.85)
    noise = rng.normal(0, 3.2, blurred.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 68])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), deepcopy(units)


def low_contrast_color_cast(image: np.ndarray, units: list[dict[str, Any]]):
    data = image.astype(np.float32)
    # Simulate a warm display screenshot with reduced contrast and a soft
    # brightness gradient from an overlay/window shadow.
    data = data * 0.68 + np.array([42, 48, 57], dtype=np.float32)
    gradient = np.linspace(0.90, 1.04, image.shape[1], dtype=np.float32)[None, :, None]
    data *= gradient
    return np.clip(data, 0, 255).astype(np.uint8), deepcopy(units)


VARIANTS: dict[str, Transform] = {
    "clean": clean,
    "scaled_085": mildly_scaled_screenshot,
    "scaled_067": scaled_screenshot,
    "jpeg_q72": jpeg_screenshot,
    "messenger_jpeg_q42": messenger_jpeg,
    "blur_noise_jpeg": blur_and_noise,
    "low_contrast_warm": low_contrast_color_cast,
}


def generate_noisy(output_dir: Path = Path("testdata/noisy")) -> Path:
    dense_dir = Path("testdata/dense")
    truth = json.loads(generate_dense(dense_dir).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for page_index, page in enumerate(truth["cases"], start=1):
        image = cv2.imread(str(dense_dir / page["image"]))
        for variant_name, transform in VARIANTS.items():
            transformed, units = transform(image, page["units"])
            filename = f"{page['name']}__{variant_name}.png"
            cv2.imwrite(str(output_dir / filename), transformed)
            cases.append({
                "page": page_index, "document": page["name"],
                "variant": variant_name, "image": filename,
                "width": transformed.shape[1], "height": transformed.shape[0],
                "units": units,
            })
    manifest = {
        "description": "Screenshot and capture degradations with transformed DOM truth",
        "variants": list(VARIANTS), "cases": cases,
    }
    path = output_dir / "ground_truth.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(generate_noisy())
