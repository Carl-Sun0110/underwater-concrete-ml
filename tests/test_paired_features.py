"""不能用大组或目标变化伪造特征增益。"""
import unittest
from concrete_audit.paired_features import paired_error_changes


class PairedFeatureTests(unittest.TestCase):
    def rows(self, predictions):
        return [{'sample_id': sid, 'actual_provisional_MPa': 10., 'oof_prediction_MPa': value}
                for sid, value in zip(['a', 'b', 'c'], predictions)]

    def test_id_pairing_and_group_weighted_effect_are_distinct(self):
        result = paired_error_changes(self.rows([0, 0, 10]), list(reversed(self.rows([5, 5, 0]))), {'a': 'A', 'b': 'A', 'c': 'B'})
        self.assertAlmostEqual(result['delta_sample_MAE_MPa'], 0.)
        self.assertAlmostEqual(result['delta_source_macro_MAE_MPa'], 2.5)
        self.assertEqual(result['source_groups_improved'], 1)
        self.assertEqual(result['source_groups_worsened'], 1)

    def test_duplicate_identity_or_changed_target_rejected(self):
        before = self.rows([0, 0, 10])
        for after in [before[:2], before + before[:1], [before[0] | {'actual_provisional_MPa': 11}, *before[1:]]]:
            with self.assertRaises(ValueError):
                paired_error_changes(before, after, {'a': 'A', 'b': 'A', 'c': 'B'})
