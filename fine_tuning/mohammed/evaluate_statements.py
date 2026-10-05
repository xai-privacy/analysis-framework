#!/usr/bin/env python3
"""Evaluate statement-level base or LoRA model on validation IDs.

Reports statement accuracy and exact-question accuracy. Choices ①..⑤ are removed
from the model input. Gold labels are derived outside the model.
"""
from __future__ import annotations
import argparse,json,re,time
from pathlib import Path
from collections import defaultdict,Counter
from mlx_lm import load,generate
from leet_statement_utils import *

SYSTEM_STATEMENT = """You are solving a LEET-Arg reasoning question.
Use only the information in the passage and statements.
Do not guess from speaker names or answer positions.
For every statement, output exactly one verdict: CORRECT or INCORRECT.
Do not output an answer-choice number."""
SYSTEM_COMPACT = """You are solving a LEET-Arg reasoning question.
Use only the information in the passage and statements.
Do not guess from speaker names or answer positions.
Evaluate every statement as CORRECT or INCORRECT, then list the statement numbers that are correct.
Do not output an answer-choice number."""

def parse_verdicts(text,n):
    out={}
    # preferred format: Statement 1: CORRECT
    for m in re.finditer(r'(?im)^\s*Statement\s*\(?([1-9]\d*)\)?\s*[:\-]\s*(CORRECT|INCORRECT)\b',text):
        i=int(m.group(1));
        if 1<=i<=n: out[i]=(m.group(2).upper()=='CORRECT')
    # fallback for (a) CORRECT, (b): INCORRECT
    for m in re.finditer(r'(?im)^\s*\(([a-z])\)\s*[:\-]?\s*(?:\*\*)?(CORRECT|INCORRECT)',text):
        i=ord(m.group(1).lower())-96
        if 1<=i<=n and i not in out: out[i]=(m.group(2).upper()=='CORRECT')
    return [out.get(i) for i in range(1,n+1)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True); ap.add_argument('--ids',required=True)
    ap.add_argument('--adapter-path'); ap.add_argument('--model',default='mlx-community/Llama-3.2-3B-Instruct-4bit')
    ap.add_argument('--mode',choices=['statement_only','statement_renamed','compact_structured'],default='statement_only')
    ap.add_argument('--seed',type=int,default=7); ap.add_argument('--max-tokens',type=int,default=500); ap.add_argument('--out',required=True)
    args=ap.parse_args()
    source={str(q['id']):q for q in read_json(args.source)}; ids=list(map(str,read_json(args.ids)))
    model,tok=load(args.model,adapter_path=args.adapter_path)
    records=[]
    for j,qid in enumerate(ids,1):
        q=source[qid]; gold,label_mode=derive_gold_statement_labels(q); user=strip_choice_numbers(q)
        mapping=None
        if args.mode=='statement_renamed':
            mapping=deterministic_name_map(qid,args.seed); user=rename_speakers(user,mapping)
        system=SYSTEM_COMPACT if args.mode=='compact_structured' else SYSTEM_STATEMENT
        prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],add_generation_prompt=True,tokenize=False)
        t=time.time(); output=generate(model,tok,prompt=prompt,max_tokens=args.max_tokens,verbose=False)
        pred=parse_verdicts(output,len(gold)); parsed=sum(x is not None for x in pred)
        correct=sum((p is not None and p==g) for p,g in zip(pred,gold)); exact=(pred==gold)
        rec={'id':qid,'category':q.get('category','UNKNOWN'),'gold_labels':gold,'pred_labels':pred,'statement_correct':correct,'statement_total':len(gold),'all_parsed':parsed==len(gold),'question_correct':exact,'label_mode':label_mode,'name_map':mapping,'seconds':round(time.time()-t,2),'output':output}
        records.append(rec); print(f'[{j}/{len(ids)}] {qid}: statements={correct}/{len(gold)} exact={exact} parsed={parsed}/{len(gold)}')
    st_total=sum(r['statement_total'] for r in records); st_correct=sum(r['statement_correct'] for r in records)
    summary={'model':args.model,'adapter_path':args.adapter_path,'mode':args.mode,'n_questions':len(records),'n_statements':st_total,'statement_accuracy':st_correct/st_total if st_total else 0,'question_accuracy':sum(r['question_correct'] for r in records)/len(records) if records else 0,'question_parse_rate':sum(r['all_parsed'] for r in records)/len(records) if records else 0}
    summary['predicted_label_counts']=dict(Counter('CORRECT' if p else 'INCORRECT' for r in records for p in r['pred_labels'] if p is not None))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'summary':summary,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
