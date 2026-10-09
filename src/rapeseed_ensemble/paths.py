# SPDX-License-Identifier: GPL-3.0-only
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def input_path(key, name):
    return Path(os.environ.get('RAPESEED_' + key, str(ROOT / 'data' / name)))
def output_path(name=''):
    root = Path(os.environ.get('RAPESEED_OUTPUT', str(ROOT / 'outputs')))
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    return p
