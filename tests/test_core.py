# SPDX-License-Identifier: GPL-3.0-only
import os
from pathlib import Path
import sys
import tempfile
import unittest

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('RAPESEED_OUTPUT', tempfile.mkdtemp(prefix='rapeseed-tests-'))
import numpy as np
import torch
from rapeseed_ensemble import components as c
from rapeseed_ensemble import pipeline as p

class CoreTests(unittest.TestCase):
    def test_month_major_selection(self):
        x = np.arange(2*84).reshape(2,84)
        actual = c.build_selected_features(x,[0,1,2,5,10],12,7)
        np.testing.assert_array_equal(actual,x.reshape(2,7,12)[:,:,[0,1,2,5,10]])

    def test_model_shapes(self):
        x = torch.zeros(4,7,5)
        for model in (c.TCN(5,48,3,0.4), c.BiLSTM(5,64,2,0.25)):
            model.eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(x).shape),(4,1))

    def test_weight_sum_and_negative_clipping(self):
        results = {m:{} for m in ['TCN','BiLSTM','RandomForest']}
        metrics = {'TCN':{'全部数据':{'R2':.8}},
                   'BiLSTM':{'全部数据':{'R2':.7}},
                   'RandomForest':{'全部数据':{'R2':-.2}}}
        w = c.compute_ensemble_weights(results,metrics,k=3)['全部数据']
        self.assertAlmostEqual(sum(w.values()),1.0)
        self.assertEqual(w['RandomForest'],0)

    def test_short_inner_training(self):
        c.set_seed(42)
        rng = np.random.RandomState(42)
        x = rng.normal(size=(30,7,5)).astype(np.float32)
        y = rng.normal(size=30).astype(np.float32)
        model = c.TCN(5,8,3,0.0)
        p.train_nn_noleak(model,x,y,epochs=2,patience=1,device=torch.device('cpu'))
        predictions = c.predict_nn(model,x[:3],torch.device('cpu'))
        self.assertTrue(np.isfinite(predictions).all())

    def test_actual_defaults(self):
        self.assertEqual(p.SEEDS,[42,0,1,2,3])
        self.assertEqual(p.TARGET_VARS,5)
        self.assertEqual(p.MONTHS,['nov','dec','jan','feb','mar','apr','May'])

    def test_small_loyo_interface(self):
        import pandas as pd
        rng = np.random.RandomState(1)
        data = pd.DataFrame(rng.normal(size=(36,84)),columns=c.FEAT_COLS)
        data['Year'] = np.repeat([2014,2015,2016],12)
        data['yield'] = rng.normal(.12,.01,36)
        data['location'] = ['SYNTHETIC_%02d' % (i%12) for i in range(36)]
        for name in ['is_drought','is_freeze','is_waterlog']:
            data[name] = np.zeros(36,dtype=int)
        original = p.train_nn_noleak
        def short_training(*args,**kwargs):
            kwargs.update(epochs=2,patience=1)
            return original(*args,**kwargs)
        p.train_nn_noleak = short_training
        try:
            result = p.loyo_cv_improved(data,torch.device('cpu'),[0,1,2,5,10],
                                       {'n_estimators':8,'max_depth':3},[42],.15)
        finally:
            p.train_nn_noleak = original
        for model in result.values():
            self.assertEqual(len(model['preds']),36)
            self.assertEqual(set(model['years']),{2014,2015,2016})
            self.assertTrue(np.isfinite(model['preds']).all())

if __name__ == '__main__':
    unittest.main()
