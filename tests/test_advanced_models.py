"""检查实际Dropout、确定性训练、折内目标缩放及CPU XGBoost可运行。"""
import importlib.util
import unittest
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from concrete_audit.advanced_models import TorchDropoutRegressor
from concrete_audit.train_compare import make_estimator

TORCH=importlib.util.find_spec('torch') is not None
XGB=importlib.util.find_spec('xgboost') is not None

@unittest.skipUnless(TORCH,'需在独立建模虚拟环境运行')
class TorchModelTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(2);X=rng.normal(size=(48,2));return X,X[:,0]-.5*X[:,1]

    def test_real_dropout_learning_and_eval_prediction(self):
        import torch
        X,y=self.data();model=TorchDropoutRegressor(epochs=80,dropout=.2).fit(X,y)
        self.assertTrue(any(isinstance(layer,torch.nn.Dropout) for layer in model.network_))
        self.assertLess(np.mean(model.loss_curve_[-10:]),np.mean(model.loss_curve_[:10]))
        self.assertEqual(model.n_iter_,80)
        pred=model.predict(X);np.testing.assert_array_equal(pred,model.predict(X))
        self.assertTrue(np.isfinite(pred).all());self.assertEqual(pred.shape,(48,))

    def test_cloning_and_fixed_seed_repeatability(self):
        X,y=self.data();m=TorchDropoutRegressor(epochs=10,random_state=7)
        a=clone(m).fit(X,y).predict(X);b=clone(m).fit(X,y).predict(X)
        np.testing.assert_allclose(a,b,rtol=0,atol=0)

    def test_target_scaling_only_uses_fit_data(self):
        X,y=self.data();y=10+y
        model=TransformedTargetRegressor(regressor=TorchDropoutRegressor(epochs=10),transformer=StandardScaler()).fit(X,y)
        before=model.transformer_.mean_.copy();model.predict(np.array([[100.,200.]]))
        np.testing.assert_allclose(before,[y.mean()]);np.testing.assert_array_equal(before,model.transformer_.mean_)

    def test_invalid_parameters_and_wrong_input_shape_rejected(self):
        X,y=self.data()
        with self.assertRaises(ValueError):TorchDropoutRegressor(dropout=1.,epochs=1).fit(X,y)
        model=TorchDropoutRegressor(epochs=1).fit(X,y)
        with self.assertRaises(ValueError):model.predict(np.ones((2,3)))

@unittest.skipUnless(XGB,'需安装CPU版XGBoost')
class XGBoostTests(unittest.TestCase):
    def test_cpu_pipeline_can_fit_predict(self):
        import json
        from pathlib import Path
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/advanced_train_only.json').read_text())
        model,grid=make_estimator('xgb','core_plus_computed_ratio',config)
        X=np.array([[100.,300.],[150.,300.],[200.,400.],[250.,400.]])
        model.set_params(model__n_estimators=5,model__max_depth=2).fit(X,[20.,25.,30.,35.])
        self.assertTrue(np.isfinite(model.predict(X)).all())
        self.assertEqual(model['model'].get_params()['device'],'cpu')
        self.assertEqual(model['model'].get_params()['n_jobs'],1)
