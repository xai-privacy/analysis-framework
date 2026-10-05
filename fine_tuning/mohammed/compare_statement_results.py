#!/usr/bin/env python3
"""Small comparison table for statement-level result JSON files."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('files',nargs='+'); args=ap.parse_args()
print(f"{'file':36} {'stmt_acc':>9} {'q_acc':>9} {'parse':>9} predictions")
for f in args.files:
 d=json.loads(Path(f).read_text()); s=d['summary']; print(f"{Path(f).name:36} {s.get('statement_accuracy',0):9.3f} {s.get('question_accuracy',0):9.3f} {s.get('question_parse_rate',0):9.3f} {s.get('predicted_label_counts',{})}")
