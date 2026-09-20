"""Conservative same-page glyph matching; no fitting or external service."""
from collections import defaultdict

import cv2
import numpy as np


def glyph_image(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    crop = image[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = np.median(np.concatenate((gray[0], gray[-1], gray[:,0], gray[:,-1])))
    contrast = np.abs(gray-bg)
    scale = np.percentile(contrast, 95)
    if scale < 30:
        return None
    return cv2.resize(np.clip(contrast/scale, 0, 1), (32,32), interpolation=cv2.INTER_LINEAR)


def repeated_glyph_candidates(image, units, max_distance=.15, margin=.65):
    """Transfer only from confident bold examples, with regular counterexamples.

    Match the SAME OCR character at similar pixel height. Translation-tolerant
    normalized squared differences avoid comparing unrelated glyph complexity.
    No recursive propagation: seed predictions remain fixed for the whole page.
    """
    groups = defaultdict(list)
    for i, unit in enumerate(units):
        if len(unit.text) == 1 and unit.text.isalnum():
            groups[unit.text].append(i)
    selected = {}
    for indices in groups.values():
        if len(indices) < 3:
            continue
        positives = [i for i in indices if units[i].bold and units[i].bold_confidence >= .7]
        negatives = [i for i in indices if not units[i].bold and
                     (units[i].textar or {}).get('bold', 1) < .2 and
                     (units[i].fontdna or {}).get('bold', 1) < .2]
        if not positives or not negatives:
            continue
        crops = {i: glyph_image(image, units[i].box) for i in indices}

        def distance(a, b):
            ha = units[a].box[3]-units[a].box[1]
            hb = units[b].box[3]-units[b].box[1]
            if min(ha,hb)/max(1,ha,hb) < .9 or crops[a] is None or crops[b] is None:
                return 1.0
            padded = cv2.copyMakeBorder(crops[b], 2,2,2,2, cv2.BORDER_CONSTANT, value=0)
            return float(cv2.matchTemplate(padded, crops[a], cv2.TM_SQDIFF_NORMED).min())

        for i in indices:
            if units[i].bold:
                continue
            positive_distance = min((distance(i,j) for j in positives), default=1.)
            negative_distance = min((distance(i,j) for j in negatives if i != j), default=1.)
            # A competing regular reference is required; abstain otherwise.
            if negative_distance < 1 and positive_distance < max_distance and positive_distance < margin * negative_distance:
                selected[i] = {'positive_distance': positive_distance, 'negative_distance': negative_distance}
    return selected
