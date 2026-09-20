"""Compare fixed recall operating points using frozen model predictions."""
import json
from pathlib import Path
from eval_utils import binary_metrics
from style_demo import Unit,style_key,textar_span_candidates

PROFILES={'balanced':(.25,.55,.40),'recall':(.15,.50,.32),'aggressive':(.10,.45,.28)}


def main():
    groups={}
    cases=[]
    for path in sorted(Path('.cache/bold_context').glob('*.json')):
        data=json.loads(path.read_text())
        suite=path.name.split('-')[0]
        variant=path.stem.rsplit('-',1)[1]
        raw=data['word_context']
        units=[Unit(**{k:v for k,v in u.items() if k!='truth'}) for u in raw]
        truth=[bool(u['truth']['bold']) for u in raw]
        metrics={}
        for name,args in PROFILES.items():
            selected=textar_span_candidates(units,*args)
            pairs=[(style_key(u)[0] or i in selected,t) for i,(u,t) in enumerate(zip(units,truth))]
            groups.setdefault(f'{suite}/{variant}',{}).setdefault(name,[]).extend(pairs)
            metrics[name]=binary_metrics(pairs)
        baseline=[Unit(**{k:v for k,v in u.items() if k!='truth'}) for u in data['baseline']]
        for name,word_min,char_min in [('gated55',.55,.10),('gated60',.60,.05),('gated65',.65,.05)]:
            pairs=[]
            for u,b,t in zip(units,baseline,truth):
                p=u.textar or {}
                accept=p.get('word_bold',0)>=word_min and (p.get('character_bold',0)>=char_min or (b.fontdna or {}).get('bold',0)>=.35)
                pairs.append((style_key(b)[0] or accept,t))
            groups.setdefault(f'{suite}/{variant}',{}).setdefault(name,[]).extend(pairs)
            metrics[name]=binary_metrics(pairs)
        cases.append({'case':path.stem,'metrics':metrics})
    report={'scope':'Fixed operating points on frozen word-context outputs; not independent validation of selected thresholds','groups':{g:{m:binary_metrics(p) for m,p in modes.items()} for g,modes in groups.items()},'cases':cases}
    Path('testdata/policy/recall_profiles.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['groups'],indent=2))


if __name__=='__main__':
    main()
