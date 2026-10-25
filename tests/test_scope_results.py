"""同范围同折可比性、逐记录覆盖及来源等权误差检查。"""
import copy
import unittest
from concrete_audit.scope_results import aggregate_predictions, validate_fold_layout


class ScopeResultTests(unittest.TestCase):
    def predictions(self):
        return [{'sample_id': name, 'feature_set': 'core2', 'model': 'mean', 'research_group': group,
                 'actual_provisional_MPa': 10., 'oof_prediction_MPa': pred}
                for name, group, pred in [('a', 'A', 0.), ('b', 'A', 0.), ('c', 'B', 10.)]]

    def test_source_macro_is_distinct_from_sample_weighted_error(self):
        metrics, groups = aggregate_predictions(self.predictions(), {'a': 'A', 'b': 'A', 'c': 'B'},
                                                {'a', 'b', 'c'}, {'test'}, {('core2', 'mean')})
        self.assertAlmostEqual(metrics[0]['MAE_MPa'], 20/3)
        self.assertEqual(metrics[0]['source_macro_MAE_MPa'], 5.)
        self.assertEqual(metrics[0]['worst_source_MAE_MPa'], 10.)
        self.assertEqual(len(groups), 2)

    def test_test_predictions_duplicates_or_missing_rows_rejected(self):
        for rows in [self.predictions() + self.predictions()[:1], self.predictions()[:2],
                     self.predictions() + [self.predictions()[0] | {'sample_id': 'test'}]]:
            with self.assertRaises(ValueError):
                aggregate_predictions(rows, {'a': 'A', 'b': 'A', 'c': 'B'}, {'a', 'b', 'c'}, {'test'}, {('core2', 'mean')})

    def test_identical_folds_required_with_no_group_leakage(self):
        report = {'folds': [{'validation_sample_ids': ['a'], 'train_groups': ['B'], 'validation_groups': ['A']},
                            {'validation_sample_ids': ['b'], 'train_groups': ['A'], 'validation_groups': ['B']}]}
        validate_fold_layout([report, copy.deepcopy(report)], {'a', 'b'})
        changed = copy.deepcopy(report)
        changed['folds'].reverse()
        with self.assertRaises(ValueError):
            validate_fold_layout([report, changed], {'a', 'b'})
        changed = copy.deepcopy(report)
        changed['folds'][0]['train_groups'] = ['A']
        with self.assertRaises(ValueError):
            validate_fold_layout([changed], {'a', 'b'})
        with self.assertRaises(ValueError):
            validate_fold_layout([report], {'a', 'b'}, [{'validation_sample_ids': ['b']}, {'validation_sample_ids': ['a']}])
