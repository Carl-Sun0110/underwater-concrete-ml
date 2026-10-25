"""检查分组防泄漏、缩放拟合范围和评估指标，不能只验证程序跑完。"""
import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from concrete_audit.baselines import leakage_groups,ensure_isolated,regression_metrics,grouped_folds


def row(group,pdf,water,cement,sf,y=1):
    return {'source_row':int(water),'original_values':{'Research Group':group,'PDF filename':pdf,
            'Water':water,'Cem':cement,'SF':sf},'Provisional_Converted_CS_28d':y}

class BaselineTests(unittest.TestCase):
    def test_full_mix_links_sources_without_target(self):
        rows=[row('a','a.pdf',200,400,0,10),row('b','b.pdf',200,400,0,99),row('c','c.pdf',200,400,5,10)]
        groups,links=leakage_groups(rows,['Water','Cem','SF'])
        self.assertEqual(groups[0],groups[1])
        self.assertNotEqual(groups[0],groups[2])
        rows[1]['Provisional_Converted_CS_28d']=-100
        self.assertEqual(list(groups),list(leakage_groups(rows,['Water','Cem','SF'])[0]))
        self.assertEqual(len(links),1)

    def test_same_paper_or_research_group_kept_together(self):
        rows=[row('a','p.pdf',100,200,0),row('a','q.pdf',110,210,1),row('b','q.pdf',120,220,2)]
        groups,_=leakage_groups(rows,['Water','Cem','SF'])
        self.assertEqual(len(set(groups)),1)

    def test_group_or_sample_leakage_rejected(self):
        g=np.array(['a','a','b'])
        with self.assertRaises(ValueError):ensure_isolated(g,np.array([0]),np.array([1]))
        with self.assertRaises(ValueError):ensure_isolated(g,np.array([0]),np.array([0]))
        ensure_isolated(g,np.array([0,1]),np.array([2]))

    def test_scaler_and_mean_fit_only_training_data(self):
        train=np.array([[1.,2.],[3.,4.]])
        validation=np.array([[1000.,2000.]])
        p=Pipeline([('scale',StandardScaler()),('ridge',Ridge())]).fit(train,[10.,20.])
        before=p['scale'].mean_.copy();p.predict(validation)
        np.testing.assert_allclose(before,[2.,3.]);np.testing.assert_allclose(p['scale'].mean_,before)
        d=DummyRegressor(strategy='mean').fit(train,[10.,20.])
        self.assertEqual(float(d.predict(validation)[0]),15.)

    def test_metrics_and_undefined_r2(self):
        m=regression_metrics([10,20],[12,18])
        self.assertEqual(m['MAE_MPa'],2.)
        self.assertEqual(m['MSE_MPa2'],4.)
        self.assertEqual(m['RMSE_MPa'],2.)
        self.assertAlmostEqual(m['MAPE_percent'],15.)
        self.assertIsNone(regression_metrics([10],[12])['R2'])
        self.assertIsNone(regression_metrics([10,10],[10,11])['R2'])

    def test_cv_group_count_and_target_signature_guard(self):
        X=np.ones((6,2));y=np.arange(6);g=np.array(['a','a','b','b','c','c'])
        folds=grouped_folds(X,y,g,3)
        self.assertEqual(len(folds),3)
        for t,v in folds:self.assertFalse(set(g[t])&set(g[v]))
        with self.assertRaises(ValueError):grouped_folds(X,y,g,4)
        with self.assertRaises(ValueError):leakage_groups([row('a','a.pdf',2,3,0)],['Water','Cem','CS_28d'])
