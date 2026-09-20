"""Offline ablation of same-glyph matching against saved policy predictions."""
import json
from pathlib import Path
import cv2
from bold_refinement import repeated_glyph_candidates
from eval_utils import assign_lines, binary_metrics
from style_demo import Unit, style_key


def main():
    root=Path('testdata/policy')
    manifest=json.loads((root/'ground_truth.json').read_text())
    totals={}
    for case in manifest['cases']:
        for variant in ('clean','jpeg_q55','resize_085'):
            folder=root/'results'/case['name']/variant
            result=json.loads((folder/'styles.json').read_text())
            units=[Unit(**u) for line in result['lines'] for u in line['units']]
            truth=case['units']
            if variant=='resize_085':
                img=cv2.imread(str(folder/'input.png'))
                original=cv2.imread(str(root/case['image']))
                sx,sy=img.shape[1]/original.shape[1],img.shape[0]/original.shape[0]
                truth=[dict(u,box=[round(v*(sx if j%2==0 else sy)) for j,v in enumerate(u['box'])]) for u in truth]
            ordered,_=assign_lines(truth)
            assert [u.text for u in ordered]==[u.text for u in units]
            image=cv2.imread(str(folder/'input.png'))
            chosen=repeated_glyph_candidates(image, units)
            for mode in ('baseline','glyph'):
                pairs=[(style_key(u)[0] or (mode=='glyph' and i in chosen),bool(t._truth['bold'])) for i,(u,t) in enumerate(zip(units,ordered))]
                totals.setdefault(variant,{}).setdefault(mode,[]).extend(pairs)
            added=[(i,ordered[i]._truth['bold']) for i in chosen]
            print(case['name'],variant,'added TP',sum(t for _,t in added),'FP',sum(not t for _,t in added),flush=True)
    report={g:{m:binary_metrics(p) for m,p in modes.items()} for g,modes in totals.items()}
    (root/'glyph_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
