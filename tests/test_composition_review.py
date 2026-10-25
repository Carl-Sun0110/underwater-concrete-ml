"""未知组分不变0，冲突总量不被覆盖，家族单来源不伪装跨体系证据。"""
import copy
import unittest
from concrete_audit.composition_review import complete_component_sum, derive_composition, family_confounding


class CompositionReviewTests(unittest.TestCase):
    def value(self, n, unit='kg/m3'):
        return {'value': n, 'unit': unit, 'status': 'reported_positive' if n is not None else 'source_table_dash_unresolved'}

    def test_unknown_component_prevents_complete_binder_sum(self):
        c = {'Cem': self.value(503.2), 'FA': self.value(59.2), 'SF': self.value(None)}
        self.assertIsNone(complete_component_sum(c, ['Cem', 'FA', 'SF'])[0])
        c['SF'] = self.value(29.6)
        self.assertEqual(complete_component_sum(c, ['Cem', 'FA', 'SF'])[0], 592.)
        self.assertIsNone(complete_component_sum(c, None)[0])

    def test_unit_mix_and_invalid_mass_rejected(self):
        c = {'Cem': self.value(1, 'relative_cement_mass'), 'FA': self.value(301)}
        with self.assertRaises(ValueError):
            complete_component_sum(c, ['Cem', 'FA'])
        c = {'Cem': self.value(1202), 'FA': self.value(-1)}
        with self.assertRaises(ValueError):
            complete_component_sum(c, ['Cem', 'FA'])

    def test_original_conflicting_total_and_density_unknown_preserved(self):
        recipe = {'components': {'Cem': self.value(1202), 'FA': self.value(301)},
                  'fibers': {'pva': self.value(19.2)}, 'defined_binder_components': ['Cem', 'FA'],
                  'water': self.value(605), 'printed_binder_total': 1502}
        original = copy.deepcopy(recipe)
        result = derive_composition(recipe)
        self.assertEqual(recipe, original)
        self.assertEqual(result['binder_component_sum'], 1503)
        self.assertTrue(result['binder_total_conflict'])
        self.assertEqual(result['source_printed_binder_total'], 1502)
        self.assertFalse(result['water_binder_model_admission'])
        recipe['water'] = self.value(605, 'L/m3')
        self.assertIsNone(derive_composition(recipe)['candidate_ratios']['water_over_defined_reported_binder_components'])

    def test_fiber_types_do_not_turn_other_fibers_into_zero(self):
        recipe = {'components': {'Cem': self.value(1, 'relative_cement_mass')},
                  'fibers': {'steel': self.value(.16, 'relative_cement_mass')},
                  'defined_binder_components': None, 'water': None}
        result = derive_composition(recipe)['candidate_ratios']
        self.assertEqual(result['steel_to_cement_mass_ratio'], .16)
        self.assertIsNone(result['pp_to_cement_mass_ratio'])
        self.assertIsNone(result['pva_to_cement_mass_ratio'])

    def test_one_source_per_family_becomes_unseen_in_source_validation(self):
        records = [{'sample_id': 'a', 'declared_mix_family': 'UHPC'}, {'sample_id': 'b', 'declared_mix_family': 'ECC'}]
        folds = [{'fold': 1, 'validation_sample_ids': ['a']}, {'fold': 2, 'validation_sample_ids': ['b']}]
        result = family_confounding(records, {'a': 'A', 'b': 'B'}, folds)
        self.assertTrue(result['all_families_single_group'])
        self.assertEqual(sum(f['unseen_family_validation_records'] for f in result['folds']), 2)
        self.assertFalse(result['automatic_categorical_feature_admission'])
