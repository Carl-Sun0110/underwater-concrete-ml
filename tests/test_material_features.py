"""防止掺量基准混用、密度假设、零值扩散和字段泄漏。"""
import copy
import unittest
from concrete_audit.material_features import make_views, pva_mass, validate_annotation, match_source_recipe, training_readiness, append_excel_row


class MaterialFeatureTests(unittest.TestCase):
    def annotation(self):
        return {'source_row': 7, 'specimen_id': 'SX1', 'registry_pdf_filename': 'paper.pdf',
                'paper': 'paper', 'feature': 'polymer_pct_cement', 'value': 0,
                'unit': '%', 'basis': 'cement_mass', 'explicit_absence_quote': 'no polymer addition',
                'evidence': [{'page': 1, 'quote': 'SX1 had no polymer addition'}]}

    def test_pva_uses_fiber_density_and_volume_percent(self):
        self.assertAlmostEqual(pva_mass(1.5, 1280), 19.2)
        self.assertAlmostEqual(pva_mass(1.75, 1280), 22.4)
        for args in [(1.5, None), (1.5, 0), (-1, 1280), (101, 1280), (True, 1280), (1.5, float('nan'))]:
            with self.assertRaises(ValueError):
                pva_mass(*args)

    def test_basis_mismatch_and_target_rejected(self):
        a = self.annotation()
        a['basis'] = 'binder_mass'
        with self.assertRaises(ValueError):
            validate_annotation(a, ['SX1 had no polymer addition'], 'hash')
        a = self.annotation()
        a['feature'] = 'CS_28d'
        with self.assertRaises(ValueError):
            validate_annotation(a, ['SX1 had no polymer addition'], 'hash')

    def test_zero_requires_explicit_absence_bound_to_evidence(self):
        a = self.annotation()
        validate_annotation(a, ['SX1 had no polymer addition'], 'hash')
        del a['explicit_absence_quote']
        with self.assertRaises(ValueError):
            validate_annotation(a, ['SX1 had no polymer addition'], 'hash')

    def test_mapping_does_not_spread_zero_or_read_targets(self):
        labels = [{'sample_id': f'id{n}', 'source_row': n, 'prior_reliability_status': '待核查',
                   'original_values': {'Specimen ID': 'SX1' if n == 7 else 'SX2',
                                       'PDF filename': 'paper.pdf', 'AWA': 0}} for n in (7, 8)]
        original = copy.deepcopy(labels)
        config = {'papers': {'paper': {'sha256': 'hash'}}, 'annotations': [self.annotation()]}
        split = [{'sample_id': 'id7', 'split': 'train'}, {'sample_id': 'id8', 'split': 'test'}]
        views, coverage = make_views(labels, config, split, {'paper': ['SX1 had no polymer addition']})
        self.assertEqual(labels, original)
        self.assertEqual(views[0]['candidate_values']['polymer_pct_cement'], 0)
        self.assertIsNone(views[1]['candidate_values']['polymer_pct_cement'])
        self.assertFalse(views[0]['admitted_to_model'])
        self.assertEqual(coverage['test']['features']['polymer_pct_cement']['unknown_count'], 1)
        config['annotations'][0]['specimen_id'] = 'wrong'
        with self.assertRaises(ValueError):
            make_views(labels, config, split, {'paper': ['SX1 had no polymer addition']})

    def test_incorrect_pva_conversion_rejected(self):
        a = self.annotation() | {'feature': 'pva_mass_kg_m3', 'value': 1920,
                                'unit': 'kg/m3', 'basis': 'mixture_volume',
                                'method': 'pva_volume_density',
                                'parameters': {'vol_pct': 1.5, 'density_kg_m3': 1280}}
        with self.assertRaises(ValueError):
            validate_annotation(a, ['SX1 had no polymer addition'], 'hash')

    def test_explicit_zero_sp_is_distinct_from_placeholder(self):
        line = '50% granulated 0.0 Cellulosic AWA 275 510 4.1'
        a = self.annotation() | {'feature': 'sp_volume_L_m3', 'unit': 'L/m3', 'basis': 'mixture_volume',
                                'evidence': [{'page': 1, 'quote': line}],
                                'zero_semantics': 'explicit_reported_dose', 'reported_zero_line': line,
                                'source_column': 'High-range water reducing agent L/m3'}
        del a['explicit_absence_quote']
        validate_annotation(a, [line], 'hash')
        a['source_column'] = 'not the SP column'
        with self.assertRaises(ValueError):
            validate_annotation(a, [line], 'hash')
        a['source_column'] = 'High-range water reducing agent L/m3'
        a['reported_zero_line'] = '50% granulated 1.5 AWA 0.0 510'
        a['evidence'] = [{'page': 1, 'quote': a['reported_zero_line']}]
        with self.assertRaises(ValueError):
            validate_annotation(a, [a['reported_zero_line']], 'hash')

    def test_recipe_match_is_exact_and_unique(self):
        recipes = [{'specimen_id': 'M-0.5', 'sp_L_m3': 2.1}]
        self.assertEqual(match_source_recipe('M-0.5', recipes)['sp_L_m3'], 2.1)
        for specimen, rs in [('M-0.5-other', recipes), ('M-0.5', recipes * 2)]:
            with self.assertRaises(ValueError):
                match_source_recipe(specimen, rs)

    def test_bound_recipe_wrong_sp_dose_rejected(self):
        a = self.annotation() | {'source_recipe': 'recipe', 'feature': 'sp_volume_L_m3',
                                'value': 12, 'unit': 'L/m3', 'basis': 'mixture_volume'}
        labels = [{'sample_id': 'id7', 'source_row': 7, 'prior_reliability_status': '待核查',
                   'original_values': {'Specimen ID': 'SX1', 'PDF filename': 'paper.pdf'}}]
        config = {'papers': {'paper': {'sha256': 'hash'}}, 'annotations': [a],
                  'source_recipes': [{'specimen_id': 'SX1', 'recipe_key': 'recipe', 'paper': 'paper', 'sp_L_m3': 6}]}
        with self.assertRaises(ValueError):
            make_views(labels, config, [], {'paper': ['SX1 had no polymer addition']})

    def test_readiness_missing_sources_uses_training_only(self):
        rows = [{'split_for_coverage_only': split, 'source_pdf_for_audit_only': source,
                 'candidate_values': {'polymer_pct_cement': value}} for split, source, value in
                [('train', 'known.pdf', 1.5), ('train', 'missing.pdf', None), ('test', 'test_only.pdf', 99)]]
        config = {'annotations': [{'registry_pdf_filename': 'known.pdf'}]}
        # 构造的视图也需要完整白名单列，不补任何材料数值。
        from concrete_audit.material_features import FEATURES
        for row in rows:
            row['candidate_values'] = {f: row['candidate_values'].get(f) for f in FEATURES}
        result = training_readiness(rows, config)
        self.assertEqual(result['training_records'], 2)
        self.assertEqual([r['registry_pdf_filename'] for r in result['unmatched_source_pdfs']], ['missing.pdf'])
        self.assertEqual(result['fields']['polymer_pct_cement']['candidate_records'], 1)
        self.assertFalse(result['selection_uses_test_or_target'])

    def test_pdf_control_characters_only_cleaned_in_excel_display(self):
        import openpyxl
        original = ['Source\x07quotation\nnext line', '=1+1', 0, None]
        sheet = openpyxl.Workbook().active
        append_excel_row(sheet, original)
        self.assertEqual(original[0], 'Source\x07quotation\nnext line')
        self.assertEqual(sheet['A1'].value, 'Sourcequotation\nnext line')
        self.assertEqual(sheet['B1'].data_type, 's')
        self.assertEqual(sheet['C1'].value, 0)
        self.assertIsNone(sheet['D1'].value)

    def test_external_upa_percent_cannot_be_rebound_to_cement(self):
        line = 'M1C0 0.4 0 1.13 533 829 829'
        a = {'feature': 'upa_reported_percent_water_medium', 'value': 0, 'unit': '%',
             'basis': 'water_medium_percent_definition_unknown',
             'evidence': [{'page': 1, 'quote': line}], 'reported_zero_line': line,
             'zero_semantics': 'explicit_reported_dose', 'source_column': 'UPA (%)'}
        validate_annotation(a, [line], 'hash')
        a['basis'] = 'cement_mass'
        with self.assertRaises(ValueError):
            validate_annotation(a, [line], 'hash')
