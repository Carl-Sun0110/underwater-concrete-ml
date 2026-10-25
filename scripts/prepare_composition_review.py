"""编录原表胶材/纤维信息，明确区分表内横线、未列项目和明确用量。"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def build():
    fine_path = ROOT / 'configs/fine_aggregate_ratio_evidence.json'
    fine = json.loads(fine_path.read_text())
    material_path = ROOT / 'configs/material_feature_evidence_three_papers.json'
    material = json.loads(material_path.read_text())
    priority_path = ROOT / 'configs/priority_paper_evidence.json'
    priority = {s['source_row']: s for s in json.loads(priority_path.read_text())['samples']}
    three_path = ROOT / 'configs/three_paper_field_evidence.json'
    three = {s['source_row']: s for s in json.loads(three_path.read_text())['samples']}
    annotations = []

    def value(n, unit, status='reported_positive'):
        return {'value': n, 'unit': unit, 'status': status}

    def unknown(unit, status='not_listed_in_reviewed_mix_table'):
        return value(None, unit, status)

    def excerpt(key, page, anchor, count, location):
        lines = (ROOT / 'outputs/paper-evidence-01' / (key + '.txt')).read_text().split('\f')[page - 1].splitlines()
        i = next(i for i, line in enumerate(lines) if anchor in line)
        return {'page': page, 'quote': '\n'.join(lines[i:i+count]), 'location': location}

    y_family = excerpt('Yousri', 1, 'Self-flowing underwater concrete', 3, '原文标题：self-flowing underwater concrete mixtures')
    b_family = excerpt('Brzozowski', 3, 'Composition of the underwater repair concretes', 2, 'Table1，underwater repair concretes')
    for src in fine['annotations']:
        row, paper = src['source_row'], src['paper']
        unit = src['denominator_unit']
        components = {name: unknown(unit) for name in ('Cem', 'FA', 'SF', 'GGBS')}
        components['Cem'] = value(src['denominator'], unit)
        fibers = {name: unknown(unit) for name in ('steel', 'pva', 'pp')}
        refs = list(src['evidence'])
        definition, water, printed = None, None, None
        if paper.startswith('Nasr'):
            sample = three[row]
            family = 'SPUC'
            refs += sample['claims']['CastingEnv']['evidence']
            refs += next(a['evidence'] for a in material['annotations'] if a['source_row'] == row and a['feature'] == 'material_system')
            notes = '表1未列矿物掺合料列，未据此断言所有矿物掺量为0；胶材定义保持待核。水量由原w/c可推算，但本轮不补造实测水量。'
        elif paper == 'Zhang2025':
            unit = 'relative_cement_mass'
            components = {name: value(n, unit) for name, n in [('Cem', 1.), ('FA', .06), ('SF', .18), ('GGBS', .19)]}
            fibers['steel'] = value(.16, unit)
            definition, water = ['Cem', 'FA', 'SF', 'GGBS'], value(.223, unit)
            family = 'UN-UHPC'
            refs += priority[row]['claims']['StF']['evidence']
            notes = '原文Table3按水泥质量归一化；胶材分项1+.06+.18+.19=1.43，不反推未知绝对kg/m³。'
        elif paper == 'AlSelmi2026':
            sample = priority[row]
            components['FA'] = value(sample['claims']['FA']['paper_value'], unit)
            fibers['pva'] = value(sample['claims']['SyF']['paper_value'], unit)
            definition, water, printed = ['Cem', 'FA'], value(605., unit), 1502.
            family = 'ECCUWC'
            refs += sample['claims']['SyF']['evidence']
            notes = 'Table4分项1202+301=1503，但表注总胶材1502；两者均保留。PVA片段不与钢/PP纤维等同。'
        elif paper == 'Yousri':
            tokens = src['evidence'][-1]['quote'].split()
            for name, column in [('SF', 2), ('FA', 3)]:
                token = tokens[column]
                components[name] = unknown(unit, 'source_table_dash_unresolved') if token in ('—', '–', '-') else value(float(token), unit)
            definition = ['Cem', 'FA', 'SF']
            water = value(float(tokens[4]), 'L/m3')
            family = 'self-flowing underwater concrete'
            refs += [y_family]
            notes = 'F.A.为细骨料、FA为粉煤灰；表内SF/FA横线未当0。Water原文为L/m³，不自动当kg/m³计算水胶比。'
        elif paper == 'Brzozowski':
            tokens = src['evidence'][-1]['quote'].split()
            water = value(float(tokens[2]), 'L/m3')
            if tokens[7] in ('—', '–', '-'):
                fibers['pp'] = unknown(unit, 'source_table_dash_unresolved')
            else:
                fibers['pp'] = value(float(tokens[7]), unit)
            family = 'underwater repair concrete'
            refs += [b_family]
            notes = '仅UCR2明确PP纤维0.9kg/m³；横线不当未掺。胶材未列项目保持未知，Water保留L/m³。'
        else:
            raise ValueError('出现未编录来源')
        annotations.append({k: src[k] for k in ('source_row', 'specimen_id', 'registry_pdf_filename', 'paper')} | {
            'components': components, 'fibers': fibers, 'defined_binder_components': definition,
            'water': water, 'printed_binder_total': printed, 'declared_mix_family': family,
            'evidence': refs, 'notes': notes, 'human_confirmation': '待核查'})
    config = {'papers': fine['papers'], 'annotations': annotations,
              'protected_inputs': list(map(str, [fine_path, material_path, priority_path, three_path])),
              'scope': '严格训练128条的组成诊断；非最终可信配比，不新增模型训练'}
    with (ROOT / 'configs/composition_review_evidence.json').open('x') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    print(f'已编录{len(annotations)}条胶材/纤维组成候选。')


if __name__ == '__main__':
    build()
