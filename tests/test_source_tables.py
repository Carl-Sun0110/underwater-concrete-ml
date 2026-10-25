"""核查时不能混用空气、7d、相对强度和不同实验处理分支。"""
import unittest
from concrete_audit.source_tables import case_rows, nasr_identity, nasr_strength, assaad_protocol, assaad_strength


class SourceTableTests(unittest.TestCase):
    def test_nasr_target_column_is_age_and_environment_bound(self):
        values = [22.47, 23.27, 33.66, 66.74, 69.13, 34.57, 35.28, 45.39, 76.15, 77.73]
        self.assertEqual(nasr_strength(values, 'FW'), 34.57)
        self.assertEqual(nasr_strength(values, 'SW'), 35.28)
        self.assertEqual(nasr_strength(values, 'SW', 7), 23.27)
        for env, age in [('Air', 28), ('FW', 90)]:
            with self.assertRaises(ValueError):
                nasr_strength(values, env, age)

    def test_complete_specimen_identity_required(self):
        self.assertEqual(nasr_identity('M1C2V3-SW', True), ('M1C2V3', 'SW'))
        for name in ['M1C2V3', 'M1C2V3-FW-extra', 'M1C2V5-FW', 'M1C2V3-Air']:
            with self.assertRaises(ValueError):
                nasr_identity(name, True)

    def test_table_parser_requires_exact_recipe_set_and_columns(self):
        good = 'M1C0 0.4 0 1.13\n5M2C1 0.4 0.01 1.23'
        rows = case_rows(good, 3, {'M1C0', 'M2C1'}, {'5M2C1': 'M2C1'})
        self.assertEqual(rows['M2C1']['printed_id'], '5M2C1')
        for bad in ['M1C0 0.4 0', 'M1C0 0.4 0 nan', 'M1C0 0.4 0 1.13\nM1C0 0.4 0 1.13']:
            with self.assertRaises(ValueError):
                case_rows(bad, 3, {'M1C0'})
        with self.assertRaises(ValueError):
            case_rows(good, 3, {'M1C0', 'M2C1'})

    def test_exposure_protocol_not_inferred_from_target(self):
        values = [42, 7.1, 37, 11.9, 5.1, 38.6, 8.1, 20, 80]
        self.assertEqual(assaad_strength(values, assaad_protocol('CRD C61 washout (plunge) test')), 37)
        self.assertEqual(assaad_strength(values, assaad_protocol('Water infiltration simulated in cell')), 38.6)
        for text in ['', 'Underwater', 'CRD C61 washout (plunge) Water infiltration simulated']:
            with self.assertRaises(ValueError):
                assaad_protocol(text)
