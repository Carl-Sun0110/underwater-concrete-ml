"""从既有原文编录严格训练部分的细骨料/水泥比，不用原库未核实绝对量推定。"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.evidence_features import positive_table_columns


def build():
    material_path = ROOT / 'configs/material_feature_evidence_three_papers.json'
    material = json.loads(material_path.read_text())
    three_path = ROOT / 'configs/three_paper_field_evidence.json'
    priority_path = ROOT / 'configs/priority_paper_evidence.json'
    three, priority = [json.loads(p.read_text()) for p in [three_path, priority_path]]
    views_path = ROOT / 'outputs/casting-scope-gate-v1/scope_views.json'
    ids = json.loads(views_path.read_text())['strict_evidence_scope']['train_sample_ids']
    labels = [r for r in json.loads((ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json').read_text()) if r['sample_id'] in set(ids)]
    annotations = []

    def add(src, paper, numerator, denominator, unit, refs, note):
        raw = src['original_values']
        annotations.append({'source_row': src['source_row'], 'specimen_id': raw['Specimen ID'],
                            'registry_pdf_filename': raw['PDF filename'], 'paper': paper,
                            'numerator': numerator, 'denominator': denominator,
                            'numerator_unit': unit, 'denominator_unit': unit,
                            'numerator_definition': '原文报告的细骨料/砂质量，不推断具体河砂或石英砂类型等同',
                            'denominator_definition': '水泥质量，不是全部胶凝材料质量',
                            'evidence': refs, 'note': note, 'human_confirmation': '待核查'})

    def get_table(paper, expected, last_id, columns):
        pages = (ROOT / 'outputs/paper-evidence-01' / (paper + '.txt')).read_text().split('\f')
        page = 3
        lines = pages[page - 1].splitlines()
        table_name = 'Table 2.' if paper == 'Yousri' else 'Table 1'
        start = next(i for i, line in enumerate(lines) if line.strip().startswith(table_name))
        end = next(i for i in range(start, len(lines)) if lines[i].split() and
                   lines[i].split()[0].replace('–', '-').replace('−', '-') == last_id)
        block = '\n'.join(lines[start:end + 1])
        values = positive_table_columns(block, expected, columns)
        first = next(i for i in range(start, end + 1) if lines[i] == next(iter(values.values()))['source_line'])
        header = {'page': page, 'location': table_name + '，水泥与细骨料质量列单位', 'quote': '\n'.join(lines[start:first])}
        return values, header

    y_ids = {r['specimen_id'] for r in material['source_recipes'] if r['paper'] == 'Yousri'}
    y_values, y_header = get_table('Yousri', y_ids, 'M-0.5-15SF', [1, 6])
    b_values, b_header = get_table('Brzozowski', {'UCR1', 'UCR2', 'UCR3'}, 'UCR3', [1, 3])
    by_three = {r['source_row']: r for r in three['samples']}
    by_priority = {r['source_row']: r for r in priority['samples']}
    for src in labels:
        row = src['source_row']
        if row in by_three:
            sample = by_three[row]
            if not sample['paper'].startswith('Nasr'):
                raise ValueError('严格训练范围不应含Assaad')
            sand, cem = sample['claims']['RS'], sample['claims']['Cem']
            if sand['unit'] != 'kg/m3' or cem['unit'] != 'kg/m3':
                raise ValueError('Nasr原文质量单位不一致')
            add(src, sample['paper'], sand['paper_value'], cem['paper_value'], 'kg/m3',
                sand['evidence'] + cem['evidence'], '只用原文表1的砂和水泥质量，不使用UPA或SP换算。')
        elif row in by_priority:
            sample = by_priority[row]
            if sample['paper'] == 'Zhang2025':
                refs = sample['claims']['SS']['evidence'] + sample['claims']['Cem']['evidence']
                add(src, sample['paper'], 1.05, 1., 'relative_cement_mass', refs,
                    'Table3已按水泥质量归一化，Sand/Cement=1.05；不使用尚未知来源的kg/m³基准或混凝土密度。')
            else:
                sand, cem = sample['claims']['RS'], sample['claims']['Cem']
                add(src, sample['paper'], sand['paper_value'], cem['paper_value'], 'kg/m3',
                    sand['evidence'] + cem['evidence'], 'Table4的S为细砂，使用564/1202；不需要解决胶材总量1502/1503矛盾。')
        else:
            raw = src['original_values']
            if raw['PDF filename'].startswith('2008-K. M. Yousri'):
                v = y_values[raw['Specimen ID']]
                add(src, 'Yousri', v['values'][1], v['values'][0], 'kg/m3',
                    [y_header, {'page': 3, 'location': 'Table2，F.A. 0–5mm是细骨料，非FA粉煤灰', 'quote': v['source_line']}],
                    '读取Cement与F.A.两列；其他列横线保持未解释，不转0。')
            elif raw['PDF filename'].startswith('2021-Piotr Brzozowski'):
                v = b_values[raw['Specimen ID'].split()[0]]
                add(src, 'Brzozowski', v['values'][1], v['values'][0], 'kg/m3',
                    [b_header, {'page': 3, 'location': 'Table1，Sand与CEM质量列', 'quote': v['source_line']}],
                    '754/429；不使用相邻Water或SP的L/m³列。')
            else:
                raise ValueError('严格训练中出现未编录质量比的来源')
    config = {'papers': material['papers'], 'annotations': annotations,
              'protected_inputs': list(map(str, [material_path, three_path, priority_path, views_path])),
              'feature': 'fine_aggregate_to_cement_mass_ratio',
              'scope': '仅固定严格训练128条，不读取测试强度，不新增可信资格；本步不训练模型'}
    with (ROOT / 'configs/fine_aggregate_ratio_evidence.json').open('x') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    print(f'已为{len(annotations)}条严格训练记录编录原文质量比候选。')


if __name__ == '__main__':
    build()
