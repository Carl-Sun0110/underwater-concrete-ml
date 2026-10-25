"""配比比值不伪造输入，内层分组与插补不接触外层验证数据。"""
import copy
import unittest
import numpy as np
from concrete_audit.nested_exploration import derive_ratios, load_development, tune


class NestedExplorationTests(unittest.TestCase):
    def setUp(self):
        self.config={'ratio_numerators':['Water','SF'],'ratio_denominator':'Cem'}

    def test_ratios_use_only_observed_positive_values(self):
        rows=[{'original_values':{'Water':180,'Cem':400,'SF':0}},
              {'original_values':{'Water':180,'Cem':0,'SF':20}},
              {'original_values':{'Water':None,'Cem':400,'SF':40}}]
        original=copy.deepcopy(rows)
        X,names=derive_ratios(rows,['Water','Cem'],self.config)
        self.assertEqual(names,['Water','Cem','Water_over_Cem','SF_over_Cem'])
        self.assertAlmostEqual(X[0,2],.45)
        self.assertTrue(np.isnan(X[0,3]));self.assertTrue(np.isnan(X[1,2:]).all())
        self.assertTrue(np.isnan(X[2,2]));self.assertEqual(X[2,3],.1)
        self.assertEqual(rows,original)

    def test_target_cannot_be_a_ratio_input(self):
        c={**self.config,'ratio_numerators':['CS_28d']}
        with self.assertRaises(ValueError):derive_ratios([],['Water'],c)

    def test_denominator_not_arbitrary_target_or_post_measurement(self):
        c={**self.config,'ratio_denominator':'Converted CS_28d'}
        with self.assertRaises(ValueError):derive_ratios([],['Water'],c)

    def test_inner_search_group_isolation_and_train_only_imputation(self):
        X=np.array([[1,np.nan],[3,np.nan],[5,np.nan],[7,np.nan],[9,np.nan],[11,np.nan]],dtype=float)
        y=np.array([10,11,12,13,14,15],dtype=float);groups=np.array(['a','a','b','b','c','c'])
        config={'inner_folds':3,'seed':42,'n_jobs':1,'selection_scoring':'neg_mean_absolute_error',
                'grids':{'rf':{'model__n_estimators':[3],'model__min_samples_leaf':[1]}}}
        search,folds=tune(X,y,groups,'rf',config)
        for t,v in folds:self.assertFalse(set(groups[t])&set(groups[v]))
        stats=search.best_estimator_['impute'].statistics_.copy()
        self.assertEqual(stats[0],6)
        self.assertTrue(np.isfinite(search.predict([[1e8,9999]])).all())
        np.testing.assert_equal(stats,search.best_estimator_['impute'].statistics_)

    def test_reuses_exact_previous_AB_cohort_and_fold_assignment(self):
        _,meta,rows,groups,y,folds=load_development('outputs/parallel-exploration-v1')
        self.assertEqual(len(rows),283)
        self.assertEqual(len(set(groups)),20)
        self.assertTrue(all(r['grade'] in {'A','B'} and r['split']=='development' for r in meta))
        self.assertEqual([len(v) for _,_,v in folds],[104,23,52,52,52])
        self.assertEqual(sorted(np.concatenate([v for _,_,v in folds]).tolist()),list(range(len(rows))))

    def test_catboost_grid_clone_and_sklearn_regression_tags(self):
        X=np.arange(24,dtype=float).reshape(12,2);y=np.arange(12)+10;groups=np.repeat(np.arange(6),2)
        config={'inner_folds':3,'seed':42,'n_jobs':1,'selection_scoring':'neg_mean_absolute_error',
                'grids':{'catboost':{'model__iterations':[4],'model__depth':[2]}}}
        search,_=tune(X,y,groups,'catboost',config)
        self.assertTrue(np.isfinite(search.predict(X[:2])).all())


if __name__=='__main__':unittest.main()
