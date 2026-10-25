"""来源等权与文献单位版本改变输入，不改变标签/样本或用验证组选参。"""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from concrete_audit.source_weighting import selection_scores, source_ratios, versioned_inputs, paired_search
from concrete_audit.nested_exploration import load_development


class SourceWeightingTests(unittest.TestCase):
    def test_macro_gives_each_source_one_vote(self):
        groups=np.array(['a']*8+['b','c']);y=np.zeros(10);pred=np.array([1.]*8+[9.,1.])
        folds=[(np.arange(8,10),np.arange(8)),(np.arange(8),np.arange(8,10))]
        s=selection_scores(y,pred,groups,folds)
        self.assertAlmostEqual(s['pooled_sample'],1.8)
        self.assertAlmostEqual(s['fold_mean'],3.)
        self.assertAlmostEqual(s['source_macro'],11/3)

    def test_invalid_inner_predictions_are_not_ignored(self):
        with self.assertRaises(ValueError):selection_scores(np.array([1]),np.array([np.nan]),np.array(['a']),[(np.array([],int),np.array([0]))])

    def test_evidence_masks_38_cells_on_18_rows_without_changing_raw_or_y(self):
        previous,_,rows,_,_,_=load_development('outputs/parallel-exploration-v1');original=copy.deepcopy(rows)
        evidence=json.loads(Path('configs/composition_review_evidence.json').read_text())
        sand=json.loads(Path('configs/fine_aggregate_ratio_evidence.json').read_text())
        versions,changes=versioned_inputs(rows,previous['config']['feature_sets']['materials'],evidence,sand)
        self.assertEqual(len(changes),38);self.assertEqual(len({r['source_row'] for r in changes}),18)
        self.assertEqual(rows,original)
        self.assertEqual(versions['recorded'][0].shape,(283,12))
        self.assertEqual(versions['unit_mask_plus_source_ratios'][0].shape,(283,18))
        i=next(i for i,r in enumerate(rows) if r['source_row']==942)
        self.assertTrue(np.isnan(versions['unit_mask'][0][i,0]))
        self.assertAlmostEqual(versions['unit_mask_plus_source_ratios'][0][i,12],.223)
        self.assertAlmostEqual(versions['unit_mask_plus_source_ratios'][0][i,17],.223/1.43)

    def test_unknown_components_and_liters_do_not_make_mass_ratio(self):
        evidence=json.loads(Path('configs/composition_review_evidence.json').read_text())
        sand=json.loads(Path('configs/fine_aggregate_ratio_evidence.json').read_text())
        r=[{'source_row':197},{'source_row':544},{'source_row':1077}]
        ratios=source_ratios(r,evidence['annotations'],sand['annotations'])
        self.assertTrue(np.isnan(ratios[0,0])) # 原文L/m³不能直接除kg/m³
        self.assertTrue(np.isnan(ratios[1,5])) # 未报告矿物成分不按0求胶凝总量
        self.assertAlmostEqual(ratios[2,5],605/1503) # 明确按分项定义，冲突仍存档

    def test_two_objectives_share_inner_fits_and_groups_stay_isolated(self):
        X=np.arange(24,dtype=float).reshape(12,2);y=np.arange(12)+10.;groups=np.repeat(np.arange(6),2)
        config={'inner_folds':3,'seed':42,'objectives':['fold_mean','source_macro'],
                'grids':{'rf':{'model__n_estimators':[3],'model__min_samples_leaf':[1,2]}}}
        _,folds,candidates,chosen=paired_search(X,y,groups,'rf',config)
        self.assertEqual(len(candidates),2);self.assertEqual(set(chosen),{'fold_mean','source_macro'})
        for t,v in folds:self.assertFalse(set(groups[t])&set(groups[v]))
        for objective in chosen:self.assertEqual(chosen[objective]['inner_scores'][objective],min(c['inner_scores'][objective] for c in candidates))


if __name__=='__main__':unittest.main()
