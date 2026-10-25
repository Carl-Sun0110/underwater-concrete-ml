import unittest
import numpy as np
from concrete_audit.scaled_comparison import filtered_search


class ConstantColumnTests(unittest.TestCase):
    def test_constant_unlearned_input_cannot_drive_predictions(self):
        X=np.column_stack([np.arange(1,13),np.ones(12)*100])
        y=np.arange(12,dtype=float)+10;groups=np.repeat(np.arange(6),2)
        config={'inner_folds':3,'seed':42,'selection_scoring':'neg_mean_absolute_error',
            'mlp_epochs':2,'mlp_dropout':.15,'grids':{
                'svr':{'model__regressor__C':[1.],'model__regressor__gamma':[.1]},
                'mlp':{'model__regressor__hidden_sizes':[[4]],'model__regressor__weight_decay':[.01]}}}
        for model in ['svr','mlp']:
            with self.subTest(model=model):
                search,_=filtered_search(X,y,groups,model,config)
                np.testing.assert_equal(search.best_estimator_['variance'].get_support(),[True,False])
                a=search.predict([[6,100]]);b=search.predict([[6,1e9]])
                np.testing.assert_allclose(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
