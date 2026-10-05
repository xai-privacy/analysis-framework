#!/usr/bin/env python3
"""Materialize an MLX saved adapter checkpoint into its own adapter folder."""
import argparse,shutil
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--adapter-dir',required=True); ap.add_argument('--iteration',required=True,type=int); ap.add_argument('--out',required=True); args=ap.parse_args()
src=Path(args.adapter_dir); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
ck=src/f'{args.iteration:07d}_adapters.safetensors'
if not ck.exists(): raise SystemExit(f'Not found: {ck}')
conf=src/'adapter_config.json'
if not conf.exists(): raise SystemExit(f'Not found: {conf}')
shutil.copy2(conf,out/'adapter_config.json'); shutil.copy2(ck,out/'adapters.safetensors'); print(out)
