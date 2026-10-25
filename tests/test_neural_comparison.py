"""检验输入/目标缩放隔离、SVR和带Dropout网络的嵌套入口。"""
import unittest
import numpy as np
from concrete_audit.neural_comparison import search_model


class NeuralComparisonTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[1,np.nan],[3,np.nan],[5,np.nan],[7,np.nan],[9,np.nan],[11,np.nan]])
        self.y=np.array([10.,12.,14.,16.,18.,20.]);self.groups=np.repeat(['a','b','c'],2)
        self.config={'inner_folds':3,'seed':42,'selection_scoring':'neg_mean_absolute_error',
            'mlp_epochs':2,'mlp_dropout':.15,'grids':{
                'svr':{'model__regressor__C':[1.],'model__regressor__gamma':[.1]},
                'mlp':{'model__regressor__hidden_sizes':[[4]],'model__regressor__weight_decay':[.01]}}}

    def test_svr_input_target_scaling_only_uses_supplied_training_data(self):
        search,folds=search_model(self.X,self.y,self.groups,'svr',self.config)
        pipe=search.best_estimator_
        self.assertEqual(pipe['scale'].mean_[0],6.)
        self.assertEqual(pipe['model'].transformer_.mean_[0],15.)
        before=pipe['model'].transformer_.mean_.copy()
        self.assertTrue(np.isfinite(search.predict([[1e6,999]])).all())
        np.testing.assert_equal(before,pipe['model'].transformer_.mean_)
        for t,v in folds:self.assertFalse(set(self.groups[t])&set(self.groups[v]))

    def test_mlp_search_clones_dropout_network_and_returns_mpa(self):
        search,_=search_model(self.X,self.y,self.groups,'mlp',self.config)
        network=search.best_estimator_['model'].regressor_
        self.assertEqual(network.dropout,.15);self.assertEqual(network.epochs,2)
        self.assertEqual(network.n_iter_,2)
        self.assertTrue(np.isfinite(search.predict(self.X)).all())

    def test_search_does_not_modify_original_missing_inputs(self):
        copy=self.X.copy();search_model(self.X,self.y,self.groups,'svr',self.config)
        np.testing.assert_equal(copy,self.X)


if __name__=='__main__':unittest.main()
