"""Word-level second view for a character-box TexTAR pipeline."""
import copy
import jieba


def word_groups(units):
    groups=[]
    owners=[]
    for line in sorted({u.line_index for u in units}):
        indices=[i for i,u in enumerate(units) if u.line_index==line]
        # Break on physical gaps as well as text boundaries (table cells).
        runs=[]
        for i in indices:
            if not runs:
                runs.append([])
            if runs[-1]:
                prev=units[runs[-1][-1]]
                h=min(prev.box[3]-prev.box[1], units[i].box[3]-units[i].box[1])
                if units[i].box[0]-prev.box[2] > .65*h:
                    runs.append([])
            runs[-1].append(i)
        for run in runs:
            text=''.join(units[i].text for i in run)
            offsets=[]
            cursor=0
            for i in run:
                offsets.append((cursor,cursor+len(units[i].text),i))
                cursor+=len(units[i].text)
            cursor=0
            seen=set()
            for token in jieba.lcut(text):
                # OCR units are atomic. A tokenizer may split "SDK-v2" inside
                # one OCR box; never duplicate that box across model windows.
                members=[i for start,end,i in offsets if end>cursor and start<cursor+len(token) and i not in seen]
                cursor+=len(token)
                if not members:
                    continue
                seen.update(members)
                group=copy.copy(units[members[0]])
                group.text=''.join(units[i].text for i in members)
                group.box=[min(units[i].box[0] for i in members),min(units[i].box[1] for i in members),max(units[i].box[2] for i in members),max(units[i].box[3] for i in members)]
                groups.append(group)
                owners.append(members)
    return groups,owners


class WordContextEnsemble:
    """Keep all non-bold attributes from the original character view."""
    def __init__(self, backend, strategy='gated'):
        self.backend=backend
        if strategy not in ('gated','max'):
            raise ValueError(strategy)
        self.strategy=strategy

    def predict(self, image, units):
        chars=[dict(p) for p in self.backend.predict(image,units)]
        groups,owners=word_groups(units)
        words=self.backend.predict(image,groups)
        if len(chars)!=len(units) or len(words)!=len(groups):
            raise ValueError('TexTAR returned an incomplete context prediction')
        for group,indices,prediction in zip(groups,owners,words):
            if len(group.text)<2 or not all(c.isalnum() for c in group.text):
                continue
            for i in indices:
                chars[i]['character_bold']=chars[i]['bold']
                chars[i]['word_bold']=prediction['bold']
                if self.strategy=='max':
                    chars[i]['bold']=max(chars[i]['bold'],prediction['bold']*.85)
                else:
                    chars[i]['word_context_enabled']=1.0
        return chars


def accept_word_bold(unit):
    prediction=unit.textar or {}
    return (prediction.get('word_bold',0)>=.55 and
            (prediction.get('character_bold',0)>=.10 or
             (unit.fontdna or {}).get('bold',0)>=.35))


def bridge_bold_gaps(units):
    """One-dimensional closing of short, supported gaps, without recursion."""
    selected=set()
    for left in range(len(units)-2):
        if not units[left].bold or units[left].bold_confidence < .2:
            continue
        gap=[]
        for right in range(left+1,min(len(units),left+4)):
            prev,current=units[right-1],units[right]
            height=min(prev.box[3]-prev.box[1],current.box[3]-current.box[1])
            if current.line_index!=units[left].line_index or current.box[0]-prev.box[2]>.5*height:
                break
            if current.bold and current.bold_confidence>=.2:
                if gap and sum(len(units[i].text) for i in gap)<=2:
                    selected.update(gap)
                break
            prediction=current.textar or {}
            punctuation=bool(current.text) and all(c in '，。、；：！？（）()、' for c in current.text)
            if not (punctuation or prediction.get('word_bold',0)>=.4 or
                    prediction.get('bold',0)>=.15 or (current.fontdna or {}).get('bold',0)>=.35):
                break
            gap.append(right)
    return selected
