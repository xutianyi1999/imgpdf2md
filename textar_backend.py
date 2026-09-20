"""CPU-friendly inference adapter for the public TexTAR checkpoint.

The upstream implementation assumes CUDA and a directory-based evaluation
dataset.  This module keeps its checkpoint-compatible architecture but accepts
the OCR boxes already available in this demo.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


TEXTAR_URL = (
    "https://huggingface.co/Tex-TAR/model_tex-tar/resolve/main/"
    "TexTAR-trained.pt"
)
TEXTAR_SHA256 = "168bfefc1a231965db891218c80320b24d360ed64bb4909bf855e4a4a0a34a6d"


def _verify_checkpoint(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != TEXTAR_SHA256:
        raise ValueError(f"TexTAR checkpoint checksum mismatch: {path}")


def ensure_textar(path: Path) -> Path:
    if path.exists():
        _verify_checkpoint(path)
        return path
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with requests.get(TEXTAR_URL, stream=True, timeout=180) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    temporary.replace(path)
    _verify_checkpoint(path)
    return path


def _build_model(sequence_size: int = 125):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=256,
                batch_first=True,
                dropout=0.20,
            )
            self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=6)

        def forward(self, value):
            return self.transformer_encoder(value)

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 8
            self.scale = (256 // self.num_heads) ** -0.5
            self.qkv = nn.Linear(256, 256 * 3, bias=True)
            self.attn_drop = nn.Dropout(0.0)
            self.proj = nn.Linear(256, 256)
            self.proj_drop = nn.Dropout(0.0)

        def forward(self, value, frequencies):
            batch, count, channels = value.shape
            qkv = self.qkv(value).reshape(
                batch, count, 3, self.num_heads, channels // self.num_heads
            ).permute(2, 0, 3, 1, 4)
            query, key, val = qkv[0], qkv[1], qkv[2]
            query_complex = torch.view_as_complex(
                query.float().reshape(*query.shape[:-1], -1, 2)
            )
            key_complex = torch.view_as_complex(
                key.float().reshape(*key.shape[:-1], -1, 2)
            )
            query = torch.view_as_real(query_complex * frequencies).flatten(3).type_as(query)
            key = torch.view_as_real(key_complex * frequencies).flatten(3).type_as(key)
            attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
            result = (self.attn_drop(attention) @ val).transpose(1, 2).reshape(
                batch, count, channels
            )
            return self.proj_drop(self.proj(result))

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(256, 256)
            self.act = nn.ReLU()
            self.drop1 = nn.Dropout(0.0)
            self.norm = nn.Identity()
            self.fc2 = nn.Linear(256, 256)
            self.drop2 = nn.Dropout(0.0)

        def forward(self, value):
            value = self.drop1(self.act(self.fc1(value)))
            return self.drop2(self.fc2(self.norm(value)))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(256)
            self.attn = Attention()
            self.drop_path = nn.Identity()
            self.norm2 = nn.LayerNorm(256)
            self.mlp = Mlp()

        def forward(self, value, frequencies):
            value = value + self.attn(self.norm1(value), frequencies)
            return value + self.mlp(self.norm2(value))

    class TexTAR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model1 = resnet18(weights=None)
            self.model1.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
            self.model2 = Encoder()
            self.model3_head = self._head()
            self.model4_head = self._head()
            self.blocks = nn.ModuleList([Block(), Block()])
            self.norm = nn.LayerNorm(256, eps=1e-6)
            # Shape is checkpoint-defined; values are replaced during loading.
            self.freqs = nn.Parameter(torch.empty(2, 2, 128))

        @staticmethod
        def _head():
            return nn.Sequential(
                nn.Linear(512, 128), nn.Dropout(0.20), nn.ReLU(),
                nn.Linear(128, 64), nn.Dropout(0.20), nn.ReLU(),
                nn.Linear(64, 32), nn.Dropout(0.20), nn.ReLU(),
                nn.Linear(32, 4),
            )

        @staticmethod
        def _frequencies(freqs, x_positions, y_positions):
            frequency_x = (
                x_positions.unsqueeze(-1).unsqueeze(1) @ freqs[0].unsqueeze(-2)
            ).view(1, 2, sequence_size, 8, -1).permute(0, 1, 3, 2, 4)
            frequency_y = (
                y_positions.unsqueeze(-1).unsqueeze(1) @ freqs[1].unsqueeze(-2)
            ).view(1, 2, sequence_size, 8, -1).permute(0, 1, 3, 2, 4)
            return torch.polar(torch.ones_like(frequency_x), frequency_x + frequency_y)

        def forward(self, images, coordinates):
            features = self.model1(images).view(1, sequence_size, 256)
            encoded = self.model2(features)
            x_positions = coordinates[:, 0].view(1, sequence_size) * 60
            y_positions = coordinates[:, 1].view(1, sequence_size) * 14
            frequencies = self._frequencies(self.freqs, x_positions, y_positions)
            contextual = encoded.detach().clone()
            for index, block in enumerate(self.blocks):
                contextual = block(contextual, frequencies[:, index])
            combined = torch.cat(
                [self.norm(contextual).view(-1, 256), encoded.view(-1, 256)], dim=-1
            )
            return torch.cat(
                [self.model3_head(combined), self.model4_head(combined)], dim=-1
            )

    return TexTAR()


class TexTARBackend:
    """Run TexTAR on OCR units in deterministic, reading-order windows."""

    sequence_size = 125

    def __init__(self, checkpoint: Path) -> None:
        import torch

        self.torch = torch
        self.model = _build_model(self.sequence_size)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    @staticmethod
    def _tensor_crop(image: np.ndarray, box: list[int]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        pad_y = max(1, math.ceil((y2 - y1) * 0.10))
        crop = image[max(0, y1):min(height, y2 + pad_y), max(0, x1):min(width, x2)]
        if crop.size == 0:
            crop = np.full((8, 8, 3), 255, np.uint8)
        border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
        if float(np.median(cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY))) < 128:
            crop = 255 - crop
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (96, 128), interpolation=cv2.INTER_AREA)
        return crop.transpose(2, 0, 1).astype(np.float32) / 255.0

    def predict(self, image: np.ndarray, units: list) -> list[dict[str, float]]:
        if not units:
            return []
        torch = self.torch
        predictions: list[dict[str, float]] = []
        starts = list(range(0, len(units), self.sequence_size))
        if len(units) > self.sequence_size and starts[-1] + self.sequence_size > len(units):
            starts[-1] = len(units) - self.sequence_size
        consumed = 0
        for start in starts:
            window = units[start:start + self.sequence_size]
            first_new = max(0, consumed - start)
            crops = [self._tensor_crop(image, unit.box) for unit in window]
            centers = np.array(
                [[(u.box[0] + u.box[2]) / 2, (u.box[1] + u.box[3]) / 2] for u in window],
                dtype=np.float32,
            )
            span = np.maximum(centers.max(axis=0) - centers.min(axis=0), 1.0)
            coordinates = (centers - centers.min(axis=0)) / span
            missing = self.sequence_size - len(window)
            if missing:
                crops.extend([np.ones((3, 128, 96), np.float32)] * missing)
                coordinates = np.vstack([coordinates, np.zeros((missing, 2), np.float32)])
            images_tensor = torch.from_numpy(np.stack(crops))
            coordinates_tensor = torch.from_numpy(coordinates)
            with torch.inference_mode():
                logits = self.model(images_tensor, coordinates_tensor)
                t1 = torch.softmax(logits[:len(window), :4], dim=-1)
                t2 = torch.softmax(logits[:len(window), 4:], dim=-1)
            for bold_group, underline_group in zip(t1[first_new:], t2[first_new:]):
                predictions.append({
                    "bold": float((bold_group[1] + bold_group[3]).item()),
                    "italic": float((bold_group[2] + bold_group[3]).item()),
                    "underline": float((underline_group[1] + underline_group[3]).item()),
                    "strikethrough": float((underline_group[2] + underline_group[3]).item()),
                })
            consumed = start + len(window)
        return predictions
