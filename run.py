# SPDX-License-Identifier: GPL-3.0-only
"""Run one explicit analysis stage with configurable private input paths."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
STAGES = {
    'train': None,
    'pca': 'pca_waterlogging.py',
    'waterlogging': 'classify_waterlogging.py',
    'bootstrap': 'bootstrap_rmse.py',
    'sensitivity': 'feature_sensitivity.py',
    'summaries': 'summarize_results.py',
}
def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.example.json')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    for key, value in config.items():
        if key.startswith('_') or value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        os.environ['RAPESEED_' + key.upper()] = str(path.resolve())
    if args.stage == 'train':
        from rapeseed_ensemble.pipeline import main as train
        train()
    else:
        runpy.run_path(str(ROOT / 'scripts' / STAGES[args.stage]), run_name='__main__')
if __name__ == '__main__':
    main()
