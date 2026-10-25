import csv
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from concrete_audit.baseline_refinement import ScaledKNNImputer, load_training
from concrete_audit.predict import predict_csv


class RefinementTests(unittest.TestCase):
    def test_knn_training_scaling_preserves_observed_values_and_unknown_columns(self):
        X=np.array([[1,100,np.nan],[3,300,np.nan],[5,np.nan,np.nan]],float)
        imputer=ScaledKNNImputer(n_neighbors=1).fit(X)
        means=imputer.mean_.copy();result=imputer.transform([[1,100,np.nan],[3,np.nan,np.nan]])
        self.assertAlmostEqual(result[0,0],1);self.assertAlmostEqual(result[0,1],100)
        self.assertAlmostEqual(result[1,1],300);self.assertTrue(np.isfinite(result).all())
        imputer.transform([[1e8,1e9,4]])
        np.testing.assert_equal(means,imputer.mean_)

    def test_fixed_holdout_rows_are_not_returned_for_training(self):
        for cohort,n in [('primary_direct',117),('exploratory_underwater',320)]:
            rows,groups,split=load_training('outputs/baseline_v1',cohort)
            self.assertEqual(len(rows),n)
            self.assertFalse({r['sample_id'] for r in rows}&set(split['test_ids']))
            self.assertEqual(len(groups),n)

    def test_csv_ignores_targets_flags_extrapolation_and_never_overwrites_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pipe=Pipeline([('impute',SimpleImputer()),('model',DummyRegressor(strategy='mean'))])
            pipe.fit([[1,2],[2,3]],[10,20]);bundle={'pipeline':pipe,'features':['Water','Cem'],
                'training_ranges':{'Water':{'min':1,'max':2},'Cem':{'min':2,'max':3}}}
            joblib.dump(bundle,p/'model.joblib')
            (p/'in.csv').write_text('case_id,Water,Cem,CS_28d\na,999,2,100\nb,,,9999\n')
            original=(p/'in.csv').read_bytes();predict_csv(p/'model.joblib',p/'in.csv',p/'out.csv')
            with (p/'out.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['outside_training_range'],'Water');self.assertEqual(rows[0]['unused_columns'],'CS_28d')
            self.assertEqual(float(rows[0]['predicted_Converted_CS_28d_MPa']),15)
            self.assertEqual(rows[1]['input_status'],'all_inputs_missing')
            self.assertEqual(original,(p/'in.csv').read_bytes())
            with self.assertRaises(ValueError):predict_csv(p/'model.joblib',p/'in.csv',p/'in.csv')
            with self.assertRaises(FileExistsError):predict_csv(p/'model.joblib',p/'in.csv',p/'out.csv')


if __name__=='__main__':unittest.main()
