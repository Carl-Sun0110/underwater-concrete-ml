import unittest
import numpy as np
from concrete_audit.baseline_v1 import clusters, select_records, split_groups, model_pipeline


def row(i,paper='a.pdf',water=180,cement=400):
    return {'sample_id':str(i),'source_row':i,'original_values':{'PDF filename':paper,'Research Group':paper,
        'Specimen ID':str(i),'Water':water,'Cem':cement,'SP':2,'CastingEnv':'Underwater',
        'CS_28d':20,'Conversion Factor':.95,'Converted CS_28d':19}}


class BaselineV1Tests(unittest.TestCase):
    def setUp(self):
        self.config={'target':'Converted CS_28d','accepted_provisional_factors':[.95,.8,.76],
            'target_consistency_tolerance_MPa':.1,'seed':42,'parameters':{'trees':4,'max_depth':2,'learning_rate':.04}}

    def test_uses_stored_target_and_keeps_unreviewed_label_flag(self):
        a=row(1);a['original_values']['CS_28d']=17;a['original_values']['Converted CS_28d']=16.1
        records,selected=select_records([a],{'1':{'scope_state':'有直接水下浇筑证据_待人工确认'}},self.config)
        self.assertEqual(selected[0]['target'],16.1)
        self.assertNotEqual(selected[0]['target'],17*.95)
        self.assertTrue(records[0]['included_primary']);self.assertEqual(records[0]['formal_reliability_status'],'待核查')

    def test_missing_or_contradictory_target_is_not_filled(self):
        a,b=row(1),row(2);a['original_values']['Converted CS_28d']=None;b['original_values']['Converted CS_28d']=30
        records,selected=select_records([a,b],{},self.config)
        self.assertFalse(selected);self.assertTrue(all(r['reasons'] for r in records))

    def test_known_non_direct_is_isolated_and_unknown_scope_remains_exploratory(self):
        records,selected=select_records([row(1),row(2)],{'1':{'scope_state':'非直接水下浇筑实验_隔离'}},self.config)
        self.assertEqual([r['sample_id'] for r in selected],['2'])
        self.assertFalse(records[1]['included_primary']);self.assertTrue(records[1]['included_exploratory'])

    def test_near_duplicate_cross_paper_is_linked_without_looking_at_target(self):
        rows=[row(1),row(2,'b.pdf',water=181),row(3,'c.pdf',water=240)]
        recipes,papers,_=clusters(rows,.01)
        self.assertEqual(recipes[0],recipes[1]);self.assertEqual(papers[0],papers[1])
        self.assertNotEqual(recipes[0],recipes[2])
        rows[1]['original_values']['CS_28d']=999
        new,_,_=clusters(rows,.01);np.testing.assert_equal(new,recipes)
        t,v=split_groups(recipes,.3,42);self.assertFalse(set(recipes[t])&set(recipes[v]))

    def test_preprocessor_fits_only_training_and_handles_all_unknown_column(self):
        pipeline=model_pipeline('rf',self.config)
        pipeline.fit([[1,np.nan],[3,np.nan],[5,np.nan]],[10,20,30])
        before=pipeline['impute'].statistics_.copy()
        self.assertEqual(before[0],3)
        self.assertTrue(np.isfinite(pipeline.predict([[1000,1]])).all())
        np.testing.assert_equal(before,pipeline['impute'].statistics_)


if __name__=='__main__':unittest.main()
