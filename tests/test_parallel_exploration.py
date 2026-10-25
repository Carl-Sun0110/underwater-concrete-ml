"""核查状态不阻塞探索，但分组隔离和标签真实性仍是必要约束。"""
import unittest
import numpy as np
from concrete_audit.parallel_exploration import allocate, feature_matrix, make_pipeline, quality


def row(i, paper='p.pdf', water=100, cement=300):
    return {'sample_id':str(i),'source_row':i,'Provisional_Converted_CS_28d':19,
            'prior_reliability_status':'待核查', 'original_values':{
                'PDF filename':paper,'Research Group':paper,'CS_28d':20,'Conversion Factor':.95,
                'Water':water,'Cem':cement,'SF':0}}


class ParallelExplorationTests(unittest.TestCase):
    def test_unreviewed_source_label_can_be_B_without_formal_promotion(self):
        q=quality(row(1),{})
        self.assertEqual(q['grade'],'B')
        self.assertFalse(q['formal_reliability_promoted'])
        self.assertEqual(q['prior_reliability_status'],'待核查')

    def test_invalid_target_cannot_be_promoted_by_override(self):
        r=row(1);r['Provisional_Converted_CS_28d']=None
        self.assertEqual(quality(r,{})['grade'],'D')
        with self.assertRaises(ValueError):
            quality(r,{}, {'grade':'B','reason':'guess','evidence':['p1']})

    def test_inconsistent_conversion_is_not_imputed_or_accepted(self):
        r=row(1);r['Provisional_Converted_CS_28d']=20
        self.assertEqual(quality(r,{})['grade'],'D')

    def test_unknown_inputs_stay_in_rows_and_target_is_forbidden(self):
        r=row(1,water=0,cement=None)
        X=feature_matrix([r],['Water','Cem','SF'])
        self.assertEqual(X.shape,(1,3));self.assertTrue(np.isnan(X).all())
        self.assertEqual(feature_matrix([r],['SF'],'recorded')[0,0],0)
        with self.assertRaises(ValueError):feature_matrix([r],['CS_28d'])

    def test_missing_new_row_of_test_source_is_reserved(self):
        a,b,c=row(1),row(2,water=None),row(3,'q.pdf',water=120)
        groups,splits,_=allocate([a,b,c],{'rows':[{'sample_id':'1','split':'test'}]},['Water','Cem'])
        self.assertEqual(groups[0],groups[1])
        self.assertEqual(splits,['reserved_historical_test','reserved_historical_test','development'])

    def test_imputer_fits_training_only_and_handles_empty_column(self):
        pipe=make_pipeline('mean','median',{'seed':42,'trees':4})
        X=np.array([[1,np.nan],[3,np.nan]],dtype=float)
        pipe.fit(X,[10,20])
        mean_before=pipe['scale'].mean_.copy()
        result=pipe.predict([[1000,5]])
        np.testing.assert_equal(mean_before,pipe['scale'].mean_)
        self.assertEqual(result[0],15)
        # 验证值1000没有改变仅用1和3拟合的训练均值2。
        self.assertEqual(mean_before[0],2)

    def test_non_direct_protocol_is_C_for_separate_experiment(self):
        self.assertEqual(quality(row(1),{'scope_state':'非直接水下浇筑实验_隔离'})['grade'],'C')


if __name__=='__main__':unittest.main()
