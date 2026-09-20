"""Bundle both fixed splits, keeping invisible-weight controls explicit."""
import json
from pathlib import Path
from build_policy_review import build
from style_demo import Unit,create_result


def main():
    root=Path('testdata/policy_validation')
    manifest=json.loads((root/'ground_truth.json').read_text())
    labels={c['name']:c for c in manifest['cases']}
    reports=[]
    groups={}
    for split in ('development','holdout'):
        data=json.loads((root/f'{split}_evaluation.json').read_text())
        for case in data['cases']:
            observable=labels[case['case']]['bold_observable']
            case['bold_observable']=observable
            reports.append(case)
            if observable:
                g=groups.setdefault(f"{split}/{case['variant']}",{})
                for mode in ('before','after'):
                    counts=g.setdefault(mode,dict(tp=0,fp=0,fn=0,tn=0,runs_complete=0,runs_total=0))
                    for key in ('tp','fp','fn','tn'):counts[key]+=case['metrics'][mode]['bold'][key]
                    counts['runs_complete']+=case['bold_runs'][mode]['complete']
                    counts['runs_total']+=case['bold_runs'][mode]['total']
            folder=root/'results'/case['case']/case['variant']
            result=json.loads((folder/'styles.json').read_text())
            units=[Unit(**u) for line in result['lines'] for u in line['units']]
            _,md=create_result(units,len(result['lines']))
            (folder/'styled.md').write_text(md,encoding='utf-8')
    for group in groups.values():
        for m in group.values():
            m['precision']=m['tp']/(m['tp']+m['fp']) if m['tp']+m['fp'] else 1.
            m['recall']=m['tp']/(m['tp']+m['fn']) if m['tp']+m['fn'] else 1.
            m['f1']=2*m['precision']*m['recall']/(m['precision']+m['recall']) if m['precision']+m['recall'] else 0.
    report={'scope':'14 visually distinguishable documents plus 2 invisible-weight controls; all cases retained; observable_groups excludes controls after pixel calibration, not prediction quality', 'observable_groups':groups,'cases':reports}
    (root/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    build(root)
    paths={mode:Path('demo_output')/f'validation_{mode}'/'dense_api_evaluation.json' for mode in ('before','after')}
    if all(path.exists() for path in paths.values()):
        api={mode:json.loads(path.read_text()) for mode,path in paths.items()}
        (root/'api_evaluation.json').write_text(json.dumps({'scope':'Two new AI Studio pages; identical OCR response caches for paired rendering; CommonMark+HTML and symmetric NFKC evaluation','results':api},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
