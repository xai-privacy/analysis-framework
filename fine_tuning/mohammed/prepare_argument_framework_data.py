#!/usr/bin/env python3
"""Prepare true Premise -> Rule -> Application -> Conclusion training data.

Requires human-reviewed structured_annotations.jsonl. The script deliberately
refuses to invent this structure from the free-form benchmark rationale.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from leet_statement_utils import *
SYSTEM="""Analyze each statement systematically. For each statement give Premises, Rule, Application, and Conclusion (CORRECT or INCORRECT). Do not output an answer-choice number."""
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def target(a):
 lines=[]
 for s in a['statements']:
  lines.append(f"Statement {s['statement']}")
  for j,p in enumerate(s['premises'],1): lines.append(f"Premise {j}: {p}")
  lines.append(f"Rule: {s['rule']}"); lines.append(f"Application: {s['application']}"); lines.append(f"Conclusion: {s['conclusion'].upper()}"); lines.append('')
 return '\n'.join(lines).strip()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--split',required=True); ap.add_argument('--annotations',required=True); ap.add_argument('--out',default='./data/argument_framework'); args=ap.parse_args()
 src={str(q['id']):q for q in read_json(args.source)}; sp=read_json(args.split); anns={str(a['id']):a for a in read_jsonl(args.annotations)}; test=set(map(str,sp['test_ids']))
 for part in ('train','valid'):
  ids=list(map(str,sp[f'{part}_ids'])); missing=[i for i in ids if i not in anns]
  if missing: raise SystemExit(f'Missing human annotations for {part}: {missing[:10]} (total {len(missing)})')
  rows=[]
  for i in ids:
   if i in test: raise SystemExit('Test leakage')
   gold,_=derive_gold_statement_labels(src[i]); conclusions=[s['conclusion'].upper()=='CORRECT' for s in anns[i]['statements']]
   if conclusions!=gold: raise SystemExit(f'{i}: annotated conclusions do not match derived gold labels')
   rows.append({'id':i,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':strip_choice_numbers(src[i])},{'role':'assistant','content':target(anns[i])}]})
  write_jsonl(Path(args.out)/f'{part}.jsonl',rows)
 print('Prepared argument-framework data from human-reviewed annotations.')
if __name__=='__main__':main()
