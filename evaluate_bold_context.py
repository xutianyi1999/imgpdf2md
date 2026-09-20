"""Paired word-context ablation, entirely local and without label fitting."""
import json
import time
from dataclasses import asdict
from pathlib import Path
import cv2
import torch
from bold_context import WordContextEnsemble
from evaluate_policy import CachedModel
from eval_utils import assign_lines,binary_metrics
from style_demo import FontDNA, classify_units, ensure_fontdna, style_key, create_result
from textar_backend import TexTARBackend,ensure_textar


def main():
    torch.set_num_threads(4)
    fontdna=CachedModel(FontDNA(ensure_fontdna(Path('.cache/fontdna/glyphdna.int8.onnx'))))
    backend=CachedModel(TexTARBackend(ensure_textar(Path('.cache/textar/TexTAR-trained.pt'))))
    ensemble=WordContextEnsemble(backend, strategy='max')
    totals={}
    cases=[]
    cache=Path('.cache/bold_context')
    cache.mkdir(parents=True,exist_ok=True)
    for suite in ('dense','style_matrix','policy'):
        root=Path('testdata')/suite
        manifest=json.loads((root/'ground_truth.json').read_text())
        for case in manifest['cases']:
            original=cv2.imread(str(root/case['image']))
            for variant in (('clean','jpeg_q55','resize_085') if suite=='policy' else ('clean',)):
                image=original
                truth=case['units']
                if variant=='jpeg_q55':
                    ok,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,55])
                    assert ok
                    image=cv2.imdecode(data,cv2.IMREAD_COLOR)
                elif variant=='resize_085':
                    image=cv2.resize(image,None,fx=.85,fy=.85,interpolation=cv2.INTER_AREA)
                    sx,sy=image.shape[1]/original.shape[1],image.shape[0]/original.shape[0]
                    truth=[dict(u,box=[round(v*(sx if i%2==0 else sy)) for i,v in enumerate(u['box'])]) for u in truth]
                fontdna.cache.clear()
                backend.cache.clear()
                metrics={}
                saved={}
                elapsed={}
                for mode,model in (('baseline',backend),('word_context',ensemble)):
                    units,lines=assign_lines(truth)
                    started=time.monotonic()
                    classify_units(image,units,lines,fontdna,model)
                    elapsed[mode]=time.monotonic()-started
                    assert all(u.textar and u.fontdna and 'error' not in u.fontdna for u in units)
                    pairs=[(style_key(u)[0],bool(u._truth['bold'])) for u in units]
                    totals.setdefault(f'{suite}/{variant}',{}).setdefault(mode,[]).extend(pairs)
                    metrics[mode]=binary_metrics(pairs)
                    saved[mode]=[dict(asdict(u),truth=u._truth) for u in units]
                    if mode=='word_context' and suite=='policy':
                        dest=root/'context_results'/case['name']/variant
                        dest.mkdir(parents=True,exist_ok=True)
                        result,md=create_result(units,len(lines))
                        cv2.imwrite(str(dest/'input.png'),image)
                        (dest/'styled.md').write_text(md,encoding='utf-8')
                        (dest/'styles.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
                (cache/f'{suite}-{case["name"]}-{variant}.json').write_text(json.dumps(saved,ensure_ascii=False),encoding='utf-8')
                cases.append(dict(suite=suite,name=case['name'],variant=variant,metrics=metrics,seconds=elapsed))
                print(suite,case['name'],variant,{m:{k:round(v[k],3) for k in ('precision','recall','f1')} for m,v in metrics.items()},flush=True)
    report={'scope':'DOM-box local style ablation; context second pass reuses first-pass model cache; timings not standalone latency','groups':{g:{m:binary_metrics(p) for m,p in modes.items()} for g,modes in totals.items()},'cases':cases}
    Path('testdata/policy/context_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
