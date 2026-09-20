"""Validate frozen soft-wrap rule on new policy families and box noise."""
import argparse
import copy
import json
import time
from pathlib import Path
import cv2
import torch
from bold_context import WordContextEnsemble,wrapped_bold_candidates
from eval_utils import assign_lines,binary_metrics
from finalize_bold_recall import complete_runs
from style_demo import FontDNA,classify_units,style_key,create_result
from textar_backend import TexTARBackend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('development','holdout'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=Path('testdata/policy_validation')
    manifest=json.loads((root/'ground_truth.json').read_text())
    font=FontDNA(Path('.cache/fontdna/glyphdna.int8.onnx'))
    model=WordContextEnsemble(TexTARBackend(Path('.cache/textar/TexTAR-trained.pt')))
    totals={}
    reports=[]
    for case in manifest['cases']:
        if case['split']!=args.split:continue
        original=cv2.imread(str(root/case['image']))
        for variant in ('clean','jpeg55','box_offset'):
            image=original
            if variant=='jpeg55':
                ok,buf=cv2.imencode('.jpg',original,[cv2.IMWRITE_JPEG_QUALITY,55])
                assert ok
                image=cv2.imdecode(buf,cv2.IMREAD_COLOR)
            units,lines=assign_lines(case['units'])
            if variant=='box_offset':
                # Shift each whole line coherently to model OCR baseline error.
                for u in units:
                    dx,dy=(1,-1) if u.line_index%2 else (-1,1)
                    u.box=[max(0,u.box[0]+dx),max(0,u.box[1]+dy),min(image.shape[1],u.box[2]+dx),min(image.shape[0],u.box[3]+dy)]
            started=time.monotonic()
            classify_units(image,units,lines,font,model,soft_wrap=False)
            seconds=time.monotonic()-started
            assert all(u.textar and u.fontdna and 'error' not in u.fontdna for u in units)
            truth=[bool(u._truth['bold']) for u in units]
            metrics={}
            runs={}
            chosen=wrapped_bold_candidates(units)
            for mode in ('before','after'):
                predicted=copy.deepcopy(units)
                if mode=='after':
                    for i in chosen:
                        predicted[i].bold=True
                        predicted[i].bold_confidence=max(predicted[i].bold_confidence,.25)
                pairs=[(style_key(u)[0],t) for u,t in zip(predicted,truth)]
                totals.setdefault(variant,{}).setdefault(mode,[]).extend(pairs)
                metrics[mode]={'bold':binary_metrics(pairs)}
                complete,total=complete_runs(predicted,truth)
                runs[mode]={'complete':complete,'total':total}
            dest=root/'results'/case['name']/variant
            dest.mkdir(parents=True,exist_ok=True)
            result,md=create_result(predicted,len(lines))
            cv2.imwrite(str(dest/'input.png'),image)
            (dest/'styles.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            (dest/'styled.md').write_text(md,encoding='utf-8')
            errors=[dict(text=u.text,box=u.box,expected_bold=t,predicted_bold=style_key(u)[0]) for u,t in zip(predicted,truth) if style_key(u)[0]!=t]
            reports.append(dict(suite='policy',case=case['name'],split=args.split,variant=variant,seconds=seconds,metrics=metrics,bold_runs=runs,bold_errors=errors))
            print(case['name'],variant,{m:{k:round(v['bold'][k],3) for k in ('precision','recall')} for m,v in metrics.items()},flush=True)
    report={'scope':'New synthetic document families with frozen thresholds; DOM boxes or controlled offsets, not real OCR accuracy','groups':{g:{m:binary_metrics(p) for m,p in modes.items()} for g,modes in totals.items()},'cases':reports}
    (root/f'{args.split}_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
