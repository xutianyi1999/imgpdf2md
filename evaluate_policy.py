"""Paired regression for dense policies, including screenshot degradation.

No API calls. Both variants share identical cached local model predictions.
DOM boxes isolate style recognition from OCR; this is not end-to-end accuracy.
"""
import hashlib
import json
from pathlib import Path

import cv2
import torch

from eval_utils import assign_lines, binary_metrics
from policy_fixtures import generate_policy
from style_demo import FontDNA, classify_units, create_result, ensure_fontdna, style_key
from textar_backend import TexTARBackend, ensure_textar


class CachedModel:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def predict(self, image, *args):
        key = (image.shape, hashlib.sha256(image.tobytes()).digest(),
               tuple(tuple(unit.box) for unit in args[0]) if args else ())
        if key not in self.cache:
            self.cache[key] = self.model.predict(image, *args)
        value = self.cache[key]
        return dict(value) if isinstance(value, dict) else value


def main():
    torch.set_num_threads(4)
    root = Path("testdata/policy")
    generate_policy(root)
    fontdna = CachedModel(FontDNA(ensure_fontdna(Path(".cache/fontdna/glyphdna.int8.onnx"))))
    textar = CachedModel(TexTARBackend(ensure_textar(Path(".cache/textar/TexTAR-trained.pt"))))
    totals = {}
    reports = []
    for suite in ("dense", "style_matrix", "policy"):
        suite_root = Path("testdata") / suite
        manifest = json.loads((suite_root / "ground_truth.json").read_text())
        for case in manifest['cases']:
            original = cv2.imread(str(suite_root / case['image']))
            variants = ["clean", "jpeg_q55", "resize_085"] if suite == "policy" else ["clean"]
            for variant in variants:
                fontdna.cache.clear()
                textar.cache.clear()
                image = original
                truth = case['units']
                if variant == "jpeg_q55":
                    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 55])
                    assert ok
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                elif variant == "resize_085":
                    image = cv2.resize(image, None, fx=.85, fy=.85, interpolation=cv2.INTER_AREA)
                    sx, sy = image.shape[1] / original.shape[1], image.shape[0] / original.shape[0]
                    truth = [dict(u, box=[round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(u['box'])]) for u in truth]
                metrics = {}
                failures = []
                for name, preserve in (("before", False), ("after", True)):
                    units, lines = assign_lines(truth)
                    classify_units(image, units, lines, fontdna, textar, preserve_span_confidence=preserve)
                    if any(not u.textar or not u.fontdna or "error" in u.fontdna for u in units):
                        raise RuntimeError("Incomplete model inference: refusing to report partial accuracy")
                    predictions = [style_key(u) for u in units]
                    pairs = {style: [(bool(p[i]) if i < 2 else p[i] is not None, bool(u._truth[style])) for p, u in zip(predictions, units)] for i, style in enumerate(("bold", "underline", "colored"))}
                    metrics[name] = {k: binary_metrics(v) for k, v in pairs.items()}
                    group = totals.setdefault(f"{suite}/{variant}", {}).setdefault(name, {k: [] for k in pairs})
                    for k, v in pairs.items():
                        group[k].extend(v)
                    if name == 'after':
                        failures = [{"text": u.text, "box": u.box, "expected_bold": u._truth['bold'], "predicted_bold": p[0]} for u, p in zip(units, predictions) if p[0] != u._truth['bold']]
                        if suite == 'policy':
                            dest = root / 'results' / case['name'] / variant
                            dest.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(dest / 'input.png'), image)
                            result, markdown = create_result(units, len(lines))
                            (dest / 'styled.md').write_text(markdown, encoding='utf-8')
                            (dest / 'styles.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                reports.append({"suite": suite, "case": case['name'], "variant": variant, "characters": len(truth), "metrics": metrics, "bold_errors": failures})
                print(suite, case['name'], variant, {k: {m: round(v['bold'][m], 4) for m in ('precision', 'recall', 'f1')} for k,v in metrics.items()}, flush=True)
    report = {"scope": "Local styles with DOM boxes; paired span-confidence ablation; synthetic documents", "groups": {g: {n: {k: binary_metrics(v) for k,v in values.items()} for n,values in names.items()} for g,names in totals.items()}, "cases": reports}
    (root / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
