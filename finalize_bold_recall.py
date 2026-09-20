"""Apply the shipping gated rule to frozen predictions and export review files."""
import json
from pathlib import Path
import shutil
from bold_context import accept_word_bold,bridge_bold_gaps
from build_policy_review import build
from eval_utils import binary_metrics
from style_demo import Unit,create_result,style_key


def complete_runs(units, truth):
    complete=total=0
    run=[]
    for i,(unit,t) in enumerate(zip(units,truth)):
        if run and (not t or unit.line_index!=units[run[-1]].line_index):
            total+=1
            complete+=all(style_key(units[j])[0] for j in run)
            run=[]
        if t:
            run.append(i)
    if run:
        total+=1
        complete+=all(style_key(units[j])[0] for j in run)
    return complete,total


def main():
    root=Path('testdata/policy')
    metadata=json.loads((root/'context_evaluation.json').read_text())['cases']
    groups={}
    reports=[]
    for case in metadata:
        suite,name,variant=case['suite'],case['name'],case['variant']
        cache=Path('.cache/bold_context')/f'{suite}-{name}-{variant}.json'
        data=json.loads(cache.read_text())
        units=[Unit(**{k:v for k,v in u.items() if k!='truth'}) for u in data['baseline']]
        truth=[bool(u['truth']['bold']) for u in data['baseline']]
        metrics={}
        run_counts={}
        for mode in ('before','after'):
            if mode=='after':
                for unit,second in zip(units,data['word_context']):
                    for key in ('word_bold','character_bold'):
                        if key in second['textar']:
                            unit.textar[key]=second['textar'][key]
                    if 'word_bold' in unit.textar:
                        unit.textar['word_context_enabled']=1.
                    if accept_word_bold(unit):
                        unit.bold=True
                        unit.bold_confidence=max(unit.bold_confidence,.35)
                for i in bridge_bold_gaps(units):
                    units[i].bold=True
                    units[i].bold_confidence=max(units[i].bold_confidence,.25)
            pairs=[(style_key(u)[0],t) for u,t in zip(units,truth)]
            metrics[mode]={'bold':binary_metrics(pairs)}
            group=groups.setdefault(f'{suite}/{variant}',{}).setdefault(mode,{'pairs':[],'runs_complete':0,'runs_total':0})
            group['pairs'].extend(pairs)
            complete,total=complete_runs(units,truth)
            group['runs_complete']+=complete
            group['runs_total']+=total
            run_counts[mode]={'complete':complete,'total':total}
        failures=[{'text':u.text,'box':u.box,'expected_bold':t,'predicted_bold':style_key(u)[0]} for u,t in zip(units,truth) if style_key(u)[0]!=t]
        reports.append(dict(suite=suite,case=name,variant=variant,metrics=metrics,bold_errors=failures,bold_runs=run_counts))
        if suite=='policy':
            dest=root/'recall_results'/name/variant
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/'context_results'/name/variant/'input.png',dest/'input.png')
            result,md=create_result(units,max(u.line_index for u in units)+1)
            (dest/'styles.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            (dest/'styled.md').write_text(md,encoding='utf-8')
    summary={g:{m:dict(binary_metrics(v['pairs']),runs_complete=v['runs_complete'],runs_total=v['runs_total']) for m,v in modes.items()} for g,modes in groups.items()}
    report={'scope':'Same synthetic suites used to select operating point; DOM boxes, not independent end-to-end validation; runs are line-local','groups':summary,'cases':reports}
    (root/'recall_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    build(root,report_name='recall_evaluation.json',result_dir='recall_results',index_name='recall_review.html')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
