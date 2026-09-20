"""Select soft-wrap bridging using only the existing development cache."""
import json
from pathlib import Path
from style_demo import Unit,style_key
from bold_context import accept_word_bold,bridge_bold_gaps,wrapped_bold_candidates
from eval_utils import binary_metrics


def main():
    groups={}
    for path in sorted(Path('.cache/bold_context').glob('*.json')):
        data=json.loads(path.read_text())
        units=[Unit(**{k:v for k,v in u.items() if k!='truth'}) for u in data['baseline']]
        for u,second in zip(units,data['word_context']):
            for k in ('word_bold','character_bold'):
                if k in second['textar']:
                    u.textar[k]=second['textar'][k]
            if accept_word_bold(u):
                u.bold=True
                u.bold_confidence=max(u.bold_confidence,.35)
        for i in bridge_bold_gaps(units):
            units[i].bold=True
            units[i].bold_confidence=max(units[i].bold_confidence,.25)
        candidates=wrapped_bold_candidates(units)
        key=path.name.split('-')[0]+'/'+path.stem.rsplit('-',1)[1]
        for mode in ('before','after'):
            pairs=[(style_key(u)[0] or (mode=='after' and i in candidates),bool(t['truth']['bold'])) for i,(u,t) in enumerate(zip(units,data['baseline']))]
            groups.setdefault(key,{}).setdefault(mode,[]).extend(pairs)
    report={g:{m:binary_metrics(p) for m,p in modes.items()} for g,modes in groups.items()}
    Path('testdata/policy/wrap_development.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
