# SPDX-License-Identifier: GPL-3.0-only
"""Check the computation-only SHAP export using synthetic demo data."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('RAPESEED_OUTPUT', tempfile.mkdtemp(prefix='rapeseed-core-check-'))
from rapeseed_ensemble import components as c

class CoreReleaseTests(unittest.TestCase):
    def test_numerical_shap_export(self):
        if not c.SHAP_AVAILABLE:
            self.fail('SHAP must be installed to test this research stage.')
        data = c.load_data(str(ROOT / 'data/synthetic_demo.xlsx'))
        previous = c.RESULT_DIR
        with tempfile.TemporaryDirectory(prefix='synthetic-shap-') as directory:
            c.RESULT_DIR = directory
            try:
                c.run_shap_analysis(data, {}, [0, 1, 2, 5, 10])
                result = pd.read_csv(Path(directory) / 'shap_values.csv')
                self.assertEqual(result.shape, (len(data), 36))
                self.assertEqual(result.columns[-1], 'group')
                self.assertTrue(result.iloc[:, :-1].notna().all().all())
                self.assertEqual([p.name for p in Path(directory).iterdir()], ['shap_values.csv'])
            finally:
                c.RESULT_DIR = previous

if __name__ == '__main__':
    unittest.main()
