"""Word-level second view for a character-box TexTAR pipeline."""
import copy
import jieba
import numpy as np


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


def wrapped_bold_candidates(units):
    """Bridge at most two supported characters across a plausible soft wrap.

    Require full-width aligned lines, similar glyph height and tight leading.
    Tables, short headings and larger paragraph gaps are deliberately excluded.
    Predictions are never propagated recursively.
    """
    lines={}
    for i,u in enumerate(units):
        lines.setdefault(u.line_index,[]).append(i)
    long_lines=[indices for indices in lines.values() if len(indices)>=8]
    if len(long_lines)<3:
        return set()
    left_edge=float(np.percentile([units[ix[0]].box[0] for ix in long_lines],25))
    right_edge=float(np.percentile([units[ix[-1]].box[2] for ix in long_lines],75))
    result=set()
    for first,second in zip(list(lines.values()),list(lines.values())[1:]):
        if min(len(first),len(second))<8:
            continue
        tail,head=units[first[-1]],units[second[0]]
        h=max(1,tail.box[3]-tail.box[1])
        next_h=max(1,head.box[3]-head.box[1])
        if min(h,next_h)/max(h,next_h)<.9:
            continue
        if not (-.1*h<=head.box[1]-tail.box[3]<=.65*h):
            continue
        if abs(tail.box[2]-right_edge)>h or abs(head.box[0]-left_edge)>h:
            continue
        if any(units[b].box[0]-units[a].box[2]>.65*h for row in (first,second) for a,b in zip(row,row[1:])):
            continue
        indices=first[-3:]+second[:3]
        virtual=[]
        for j,i in enumerate(indices):
            u=copy.copy(units[i])
            u.line_index=0
            u.box=[j*int(h),0,(j+1)*int(h),int(h)]
            virtual.append(u)
        candidates=bridge_bold_gaps(virtual)
        # Only a gap touching the wrap is new evidence; within-line gaps are
        # handled separately, with their original geometry.
        if 2 in candidates or 3 in candidates:
            result.update(indices[j] for j in candidates if j in (1,2,3,4))
    return result
