"""编录新上传三篇原文：材料候选与强度/实验范围核查分开，不训练。"""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.material_features import FEATURES
from concrete_audit.source_tables import case_rows, nasr_identity, nasr_strength, assaad_protocol, assaad_strength


def build():
    folder = ROOT / 'outputs/three-paper-review-v1'
    papers = json.loads((folder / 'papers.json').read_text())
    pages = {k: (folder / (k + '.txt')).read_text().split('\f') for k in papers}
    inputs = json.loads((ROOT / 'configs/provisional_inputs.json').read_text())
    labels = json.loads((ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json').read_text())
    material = json.loads((ROOT / 'configs/material_feature_evidence_extended.json').read_text())
    material['papers'].update(papers)
    material['scope'] = '10篇已上传原文；新增3篇134条定向记录，候选不代表最终可信'
    samples, issues, protocols = [], [], []

    def ref(key, page, quote, location):
        return {'page': page, 'quote': quote, 'location': location}

    def excerpt(key, page, anchor, count, location):
        lines = pages[key][page - 1].splitlines()
        i = next(i for i, line in enumerate(lines) if anchor in line)
        return ref(key, page, '\n'.join(lines[i:i + count]), location)

    def table(key, page, name, final_id, ncol, expected, aliases=None):
        lines = pages[key][page - 1].splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip().startswith(name))
        end = next(i for i in range(start, len(lines)) if lines[i].split() and lines[i].split()[0] == final_id)
        body = '\n'.join(lines[start:end + 1])
        rows = case_rows(body, ncol, expected, aliases)
        first = next(i for i in range(start, end + 1) if lines[i] == next(iter(rows.values()))['source_line'])
        header = ref(key, page, '\n'.join(lines[start:first]), name + '，原表表头和单位')
        return rows, header

    def direct(value, unit, refs, note=''):
        return {'mode': 'direct', 'paper_value': value, 'comparison_value': value, 'unit': unit,
                'evidence': refs, 'note': note, 'action': '字段级辅助结果，仍需人工确认，不提升整条可靠状态。'}

    def pending(value, unit, refs, note):
        return {'mode': 'ambiguous', 'paper_value': value, 'unit': unit, 'evidence': refs,
                'note': note, 'action': '保留原值及疑点，禁止按算术一致自动认定可靠。'}

    def add(src, key, name, value, refs, note='', **extra):
        unit, basis = FEATURES[name]
        material['annotations'].append({'source_row': src['source_row'],
            'specimen_id': src['original_values']['Specimen ID'],
            'registry_pdf_filename': src['original_values']['PDF filename'], 'paper': key,
            'feature': name, 'value': value, 'unit': unit, 'basis': basis, 'evidence': refs,
            'method': 'curated_source_reading', 'note': note,
            'human_confirmation': '待核查', 'admitted_to_model': False, **extra})

    static_cases = {f'M{m}C{c}' for m in (1, 2, 3) for c in (0, 1, 2, 3)} | {f'M{m}C{c}' for m in (4, 5, 6, 7) for c in (0, 1)}
    moving_cases = {f'M{m}C{c}V{v}' for m in range(1, 5) for c in (1, 2) for v in range(1, 5)}
    for key, moving, expected, target_page, last in [
        ('NasrStrength2022', False, static_cases, 6, 'M7C1'),
        ('NasrCurrents2022', True, moving_cases, 7, 'M4C2V4')]:
        mixes, mh = table(key, 6, 'Table 1', last, 7 if moving else 6, expected)
        targets, th = table(key, target_page, 'Table 2', last, 10, expected,
                            {'5M2C1': 'M2C1', '6M3C1': 'M3C1'} if not moving else {})
        environment_ref = (excerpt(key, 3, 'Conventional UWC approaches', 4, '§2，UPA加入外部水中') if not moving else
                           excerpt(key, 3, 'novel AWA that behaves', 6, '§2.1，UPA加入外部水介质'))
        casting = (excerpt(key, 4, 'Two molds were then placed', 6, '§3.3.1，FW/SW中浇筑及龄期') if not moving else
                   excerpt(key, 3, 'ments during the pouring', 6, '§2.3.1，水中浇筑'))
        spec = (excerpt(key, 4, 'representative concrete sample', 4, '§3.3.1，150mm试件') if not moving else
                excerpt(key, 5, 'moulds (sizes:', 3, '§2.3.2，150mm切割立方体'))
        match_name = 'Strength evaluation-of-a-new' if not moving else 'Influence-of-water-currents'
        selected = [r for r in labels if match_name in r['original_values']['PDF filename']]
        for src in selected:
            raw = src['original_values']
            case, env = nasr_identity(raw['Specimen ID'], moving)
            wc, upa, sp, cem, sand, rock = mixes[case]['values'][:6]
            mr = [mh, ref(key, 6, mixes[case]['source_line'], f'Table1，{case}')]
            tr = [th, ref(key, target_page, targets[case]['source_line'], f'Table2，{case}，28d/{env}')]
            value = nasr_strength(targets[case]['values'], env)
            add(src, key, 'material_system', 'SPUC', [environment_ref, mh])
            add(src, key, 'exposure_protocol', 'direct_cast_into_water_medium', [casting, environment_ref])
            add(src, key, 'water_environment', {'FW': 'freshwater', 'SW': 'seawater'}[env], tr + [casting])
            add(src, key, 'water_motion_condition', 'flowing' if moving else 'stagnant',
                mr if moving else [excerpt(key, 3, 'fresh and marine stagnant water', 2, '§3，静水试验')])
            zero = {} if upa else {'zero_semantics': 'explicit_reported_dose',
                                  'reported_zero_line': mixes[case]['source_line'], 'source_column': 'UPA (%)'}
            add(src, key, 'upa_reported_percent_water_medium', upa, mr + [environment_ref],
                '仅保存原表浓度%；外部水体基准，质量/体积定义仍未知，不转为混凝土AWA kg/m³。', **zero)
            add(src, key, 'sp_volume_L_per_100kg_cement', sp, mr,
                '仅按表格L/100kg水泥保存；不以原库kg/m³或异常密度表述自动证明质量换算。')
            if moving:
                add(src, key, 'water_velocity_m_s', mixes[case]['values'][6], mr)
            claims = {'CS_28d': direct(value, 'MPa', tr, '明确取28d FW/SW绝对强度列，不取空气或相对强度列。'),
                'Cem': direct(cem, 'kg/m3', mr), 'RS': direct(sand, 'kg/m3', mr), 'CA': direct(rock, 'kg/m3', mr),
                'w/c': direct(wc, '-', mr), 'CS_Spec': direct('Cu150', '-', [spec]),
                'CastingEnv': direct('Underwater', '-', [casting]),
                'UPA_Conc': direct(upa, '%', mr + [environment_ref], '百分比为外部水体的报告浓度，质量/体积定义仍未知；不核实AWA质量。'),
                'SP': pending(sp, 'L/100kg水泥', mr, '当前原库为kg/m³；静水论文正文又写kg/100kg，密度表述亦需澄清，不能靠乘算一致判定可靠。'),
                'CastMethod': pending('direct underwater casting', '-', [casting], '方法类别已定位；不把详细原库自由文本自动判成逐字一致。')}
            if moving:
                claims['WaterVelocity'] = direct(mixes[case]['values'][6], 'm/s', mr)
            fields = ['CS_28d', 'CS_Spec', 'CastingEnv', 'Cem', 'RS', 'CA', 'w/c', 'Water', 'SP',
                      'UPA_Conc', 'CastMethod', 'Density', 'Conversion Factor', 'Converted CS_28d']
            if moving:
                fields.append('WaterVelocity')
            samples.append({'source_row': src['source_row'], 'paper': key, 'specimen_id': raw['Specimen ID'],
                            'registry_pdf_filename': raw['PDF filename'], 'fields': fields, 'claims': claims})
            protocols.append({'source_row': src['source_row'], 'paper': key, 'protocol': 'direct_cast_into_water_medium',
                              'scope_action_proposal': '浇筑方式有据，其他疑点未解决，不提升整条状态', 'status_changed': False})
        issues.append({'paper': key, 'issue': 'UPA投加位置与混凝土AWA不同；§Materials的0.05%与表1/配比段0.005%不一致',
                       'affected_source_rows': [r['source_row'] for r in selected if re.match(r'M\d+C2', r['original_values']['Specimen ID'])],
                       'action': '分别保留原表和正文证据；不改成某个推断的正确值，不转换为混凝土内掺质量',
                       'evidence': [excerpt(key, 3, 'gravities were 2.69', 5, '§3.1，正文所列UPA浓度') if not moving else environment_ref,
                                    mh, ref(key, 6, mixes['M1C2V1' if moving else 'M1C2']['source_line'], 'Table1，C2明确列0.005%')]})
        if not moving:
            issues.append({'paper': key, 'issue': 'Table1 SP为L/100kg水泥，正文§3.4写kg/100kg水泥',
                           'affected_source_rows': [r['source_row'] for r in selected],
                           'evidence': [mh, excerpt(key, 4, 'ranged from 1.13', 2, '§3.4，SP质量单位表述')],
                           'action': '表格体积值独立保留，原库SP质量值继续待核，不假定密度即可证明换算可靠'})

    key = 'Assaad2013'
    mix_ids = {str(i) for i in [1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17, 18, 19, 20, 22]}
    targets, th = table(key, 12, 'Table 5', '22', 9, mix_ids)
    # Table3明确区分CEM与UWC列；只编录UWC HRWR，不取相邻砂浆列。
    dosage = {1: (None, 1.02), 3: (.015, .83), 4: (None, .6), 6: (None, .62), 7: (None, .48),
              9: (.01, .65), 11: (None, .6), 12: (.05, .63), 14: (None, .5), 15: (None, .36),
              17: (.02, .7), 18: (None, .69), 19: (None, .37), 20: (.08, .42), 22: (None, .27)}
    table3 = excerpt(key, 7, 'Table 3 Results', 33, 'Table3，UWC与CEM列分别编录')
    material_ref = excerpt(key, 3, 'to 40 % and 1.11', 6, '§3.1，粉状welan gum与缓凝剂')
    retarder = excerpt(key, 3, 'mixing, the set-retarder', 2, '§3.2，UWC为水泥质量0.6%；CEM为0.4%')
    wash = excerpt(key, 5, 'Right after measuring W3', 16, '§3.4.3，冲洗后回收、混合、重新装圆柱')
    infiltration = excerpt(key, 5, 'drop in compressive strength due', 21, '§3.4.3，渗水后回收、重新装圆柱')
    selected = [r for r in labels if r['original_values']['PDF filename'].startswith('2013-Joseph J. Assaad')]
    for src in selected:
        raw = src['original_values']
        number = int(re.fullmatch(r'Mix (\d+)', raw['Specimen ID']).group(1))
        protocol = assaad_protocol(raw['CastMethod'])
        refs = [wash if protocol == 'washout_then_remolded' else infiltration]
        tr = [th, ref(key, 12, targets[str(number)]['source_line'], f'Table5，Mix{number}，{protocol}')]
        value = assaad_strength(targets[str(number)]['values'], protocol)
        add(src, key, 'exposure_protocol', protocol, refs, '不是直接水下浇筑；后续严格浇筑范围应隔离，当前不改历史样本。')
        awa, sp = dosage[number]
        add(src, key, 'sp_reported_pct_cement', sp, [table3], '仅按原文% of cement记录，未进一步推断百分比类型。')
        add(src, key, 'retarder_pct_cement', .6, [retarder], '原文明确UWC水泥质量基准，不能误用CEM的0.4%。')
        if awa is not None:
            add(src, key, 'awa_reported_pct_cement', awa, [table3], '原文% of cement，未把横线变0。')
            add(src, key, 'antiwashout_product', 'powdered welan gum', [table3, material_ref])
        claims = {'CS_28d': direct(value, 'MPa', tr + refs, '该值对应冲洗/渗水后重装模试验，不是直接水下浇筑强度。'),
                  'CS_Spec': direct('Cy100', '-', refs),
                  'CastingEnv': pending(protocol, '-', refs, '原文为暴露处理后回收、装模；与库中Underwater简单标签不等价，须先解决研究范围。'),
                  'SP': pending(sp, '% of cement', [table3], '只核实UWC HRWR列；不取CEM列，不把质量乘算一致当成可靠证明。'),
                  'AWA': pending(awa if awa is not None else '原表横线', '% of cement', [table3], '横线不自动转0，百分比基准定义及绝对质量单独核查。'),
                  'Retarder': {'mode': 'calculated', 'paper_value': .6, 'unit': '%水泥质量', 'evidence': [retarder, table3],
                               'formula_id': 'percent_mass', 'formula': '.6/100 * cement_kg_m3',
                               'parameters': {'percent': .6, 'mass_kg_m3': 350 if number <= 7 else 400 if number <= 15 else 450},
                               'note': '仅独立计算有据候选，原库Retarder=0不能视为未掺，禁止回写原表。'}}
        samples.append({'source_row': src['source_row'], 'paper': key, 'specimen_id': raw['Specimen ID'],
                        'registry_pdf_filename': raw['PDF filename'],
                        'fields': ['CS_28d', 'CS_Spec', 'CastingEnv', 'SP', 'AWA', 'Retarder', 'Density', 'Conversion Factor', 'Converted CS_28d'],
                        'claims': claims})
        protocols.append({'source_row': src['source_row'], 'paper': key, 'protocol': protocol,
                          'scope_action_proposal': '后续严格水下浇筑主分析先隔离；当前档案/训练划分未改变',
                          'requires_human_confirmation': True, 'status_changed': False, 'evidence': refs})
    issues.append({'paper': key, 'issue': '30条为冲洗/渗水暴露后重新装模，不能当作直接水下浇筑实验',
                   'affected_source_rows': [s['source_row'] for s in selected], 'evidence': [wash, infiltration],
                   'action': '生成范围隔离建议；不删除、不修改既有标签或可靠状态，后续训练先落实范围门禁'})
    material['report_notes'] = [n for n in material['report_notes'] if '7篇' not in n]
    material['report_notes'] += ['本轮新上传三篇134条：UPA外部水体浓度与混凝土内掺AWA分开；SP表格与正文单位矛盾保留。',
                                 'Assaad30条仅有部分字段定位，实验方式不属于直接水下浇筑，不可当作新增可信样本。',
                                 'UPA原表0只绑定明确控制配方；横线及未编录记录仍未知。']
    material['other_available_registry_filenames'] = sorted(set(material['other_available_registry_filenames']) |
                                                          {s['registry_pdf_filename'] for s in samples})
    cfg = {**{k: inputs[k] for k in ['raw', 'annotated', 'registry']},
           'raw_sha256': hashlib.sha256(Path(inputs['raw']).read_bytes()).hexdigest(), 'papers': papers,
           'samples': samples, 'scope': '134条字段级辅助核查，人工未签字，不改变状态，不训练'}
    for name, data in [('three_paper_field_evidence.json', cfg), ('material_feature_evidence_three_papers.json', material)]:
        (ROOT / 'configs' / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    for name, data in [('source_issues.json', issues), ('protocol_scope_proposals.json', protocols)]:
        (folder / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'samples': len(samples), 'material_annotations_total': len(material['annotations']),
                      'protocol_scope_flags': sum(p['paper'] == 'Assaad2013' for p in protocols)}, ensure_ascii=False))


if __name__ == '__main__':
    build()
