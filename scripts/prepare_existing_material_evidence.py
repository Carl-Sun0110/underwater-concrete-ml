"""扩展已上传原文的材料语义编录；完整来源配方ID白名单，不按论文自动填充。"""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.material_features import FEATURES, match_source_recipe


def build():
    """把逐表阅读的配方/单位对应关系保存为可审查配置，不修改旧配置。"""
    old = json.loads((ROOT / 'configs/material_feature_evidence.json').read_text())
    pilot = json.loads((ROOT / 'configs/pilot_evidence.json').read_text())
    labels = json.loads((ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json').read_text())
    pages = {k: (ROOT / 'outputs/paper-evidence-01' / (k + '.txt')).read_text().split('\f') for k in pilot['papers']}
    config = {**old, 'papers': old['papers'] | pilot['papers'], 'source_recipes': [],
              'other_available_registry_filenames': sorted({s['registry_pdf_filename'] for s in pilot['samples']}),
              'scope': '7篇PDF已具备；5篇形成当前458条内的字段候选，另外2篇未恢复入探索集',
              'report_notes': []}
    notes = config['report_notes']
    notes += ['本轮扩展Sonebi、Yousri、Brzozowski；Wang2021和Chaikasatsin在当前458条中没有记录，未恢复入集。',
              'Sonebi表3/4中的control None仅对应3条指定控制配方；不是把所有AWA=0当未掺。',
              'Sonebi液态纤维素AWA按L/100kg胶材保存；缓凝剂85mL/100kg胶材不转换成kg/m³。',
              'Yousri VA单独建列，不将VA当成welan-gum AWA或把横线当零；百分比基准为水泥质量。',
              'Brzozowski表1的SP和乳液按L/m³保留，PP纤维仅UCR2报告0.9kg/m³；异常印刷密度不采用。',
              '未对未编录来源默认标为普通混凝土，也未把同篇论文的标签自动传播到其他来源。']

    def ref(paper, page, quote, location):
        return {'page': page, 'quote': quote, 'location': location}

    def excerpt(paper, page, anchor, count, location):
        lines = pages[paper][page - 1].splitlines()
        start = next(i for i, line in enumerate(lines) if anchor in line)
        return ref(paper, page, '\n'.join(lines[start:start + count]), location)

    def add(src, paper, feature, value, refs, note='', recipe=None, **extra):
        unit, basis = FEATURES[feature]
        item = {'source_row': src['source_row'], 'specimen_id': src['original_values']['Specimen ID'],
                'registry_pdf_filename': src['original_values']['PDF filename'], 'paper': paper,
                'feature': feature, 'value': value, 'unit': unit, 'basis': basis, 'evidence': refs,
                'method': 'curated_source_reading', 'note': note,
                'human_confirmation': '待核查', 'admitted_to_model': False, **extra}
        if recipe:
            item['source_recipe'] = recipe['recipe_key']
        config['annotations'].append(item)

    # 以下每个SP序列来自Sonebi表3/4逐组阅读，含无强度值而不在458内的行。
    profiles = {
        5: [('C100-NONE', [1.3, 2, 4]), ('C100-WG0.07', [2.7, 3.5, 4.5, 5.2]),
            ('C100-WG0.15', [4.9, 7.5, 8.2]), ('SF10-WG0.07', [4.2, 5.1, 6.3, 8]),
            ('SF10-WG0.15', [5, 6.5, 7.7]), ('SF6FA20-WG0.07', [5.9, 7.2, 8.2, 9.4]),
            ('SF6FA20-WG0.15', [6, 6.9, 7.8, 8.5, 9.3]), ('SL50-WG0.07', [2.1, 2.9, 3.6, 4.3, 5.4]),
            ('SL50-WG0.15', [3.1, 3.9, 4.6, 5.4, 5.9]), ('C100-CL1.00', [1.4, 2]),
            ('SF10-CL1.00', [2.5, 3.5, 4.1]), ('SF6FA20-CL1.00', [1.9, 3.2, 3.9]),
            ('SL50-CL1.00', [0, 1.9, 2.6])],
        6: [('C100-WG0.07', [.5, .9, 1.7, 2.9]), ('C100-WG0.15', [0, .4, 1.4, 2, 3.1]),
            ('SF10-WG0.07', [3.2, 4, 5, 5.6]), ('SF10-WG0.15', [3.6, 4.1, 4.6, 5.1]),
            ('SF6FA20-WG0.07', [4, 5, 5.9, 7]), ('SL50-WG0.07', [.7, 1.2, 1.8, 2.8]),
            ('C100-CL1.00', [0, .1, .2, .25, .35, .5]), ('C100-CL1.65', [0, 1, 1.5]),
            ('SF10-CL1.00', [3.1, 3.9, 4.5]), ('SF6FA20-CL1.00', [1, 2.3, 3]),
            ('SL50-CL1.65', [0, 1, 1.7])],
    }
    recipe_refs = {}
    for page, groups in profiles.items():
        lines = pages['Sonebi'][page - 1].splitlines()
        cursor = 5
        positions = []
        for prefix, sp_values in groups:
            row_positions = []
            for sp in sp_values:
                found = None
                for i in range(cursor, len(lines)):
                    tokens = re.findall(r'(?<!\S)\d+(?:\.\d+)?(?!\S)', lines[i])
                    if tokens and float(tokens[0]) == sp:
                        found = i
                        break
                if found is None:
                    raise ValueError(f'来源表行未定位：{page}/{prefix}/{sp}')
                row_positions.append(found)
                cursor = found + 1
            positions.append((prefix, sp_values, row_positions))
        table_end = next(i for i, line in enumerate(lines) if line.startswith('Note:'))
        header = ref('Sonebi', page, '\n'.join(lines[:5]), f'Table{page - 2}，SP单位表头')
        for j, (prefix, sp_values, row_positions) in enumerate(positions):
            end = positions[j + 1][2][0] if j + 1 < len(positions) else table_end
            group_ref = ref('Sonebi', page, '\n'.join(lines[row_positions[0]:end]), f'Table{page - 2}，{prefix}配方组')
            for sp, pos in zip(sp_values, row_positions):
                w = .41 if page == 5 else .47
                specimen = f'{prefix}-SP{sp:g}-w{w:g}'
                recipe = {'paper': 'Sonebi', 'recipe_key': f'Sonebi:p{page}:{specimen}',
                          'specimen_id': specimen, 'sp_L_m3': sp, 'source_page': page,
                          'awa_code': prefix.split('-')[1], 'source_row_line': lines[pos]}
                config['source_recipes'].append(recipe)
                recipe_refs[recipe['recipe_key']] = [header, group_ref]

    son_rows = [r for r in labels if r['original_values']['PDF filename'] == pilot['samples'][0]['registry_pdf_filename']]
    son_recipes = [r for r in config['source_recipes'] if r['paper'] == 'Sonebi']
    # 表1只用于剂量单位和固定缓凝剂85，不以组分相加推算未知AWA质量。
    retarder = excerpt('Sonebi', 4, 'Set retarder,', 3, 'Table1，85mL/100kg胶材')
    son_material = excerpt('Sonebi', 3, 'A liquid-based cellulosic', 6, 'Materials，液态纤维素和粉状welan gum')
    for src in son_rows:
        recipe = match_source_recipe(src['original_values']['Specimen ID'], son_recipes)
        refs = recipe_refs[recipe['recipe_key']]
        code = recipe['awa_code']
        if code == 'NONE':
            add(src, 'Sonebi', 'awa_pct_binder', 0, refs, 'Table3指定控制组None，不外推其他零。',
                recipe=recipe, explicit_absence_quote='None')
            add(src, 'Sonebi', 'antiwashout_product', 'none (source control)', refs, recipe=recipe)
        elif code.startswith('WG'):
            add(src, 'Sonebi', 'awa_pct_binder', float(code[2:]), refs, recipe=recipe)
            add(src, 'Sonebi', 'antiwashout_product', 'powdered welan gum', refs + [son_material], recipe=recipe)
        elif code.startswith('CL'):
            add(src, 'Sonebi', 'awa_volume_L_per_100kg_binder', float(code[2:]), refs,
                '体积/胶材质量，不能当成质量百分比。', recipe=recipe)
            add(src, 'Sonebi', 'antiwashout_product', 'liquid cellulosic AWA', refs + [son_material], recipe=recipe)
        extra = {} if recipe['sp_L_m3'] else {'zero_semantics': 'explicit_reported_dose',
                    'reported_zero_line': recipe['source_row_line'], 'source_column': 'High-range water reducing agent L/m3'}
        add(src, 'Sonebi', 'sp_volume_L_m3', recipe['sp_L_m3'], refs, recipe=recipe, **extra)
        add(src, 'Sonebi', 'retarder_mL_per_100kg_binder', 85, [retarder],
            '不把原库Retarder=0解读为未掺；仅存原文体积剂量。', recipe=recipe)

    # Yousri表2各配方完整ID与百分比，特殊VA单独保留，不把表中横线变零。
    yt = excerpt('Yousri', 3, 'Table 2. Concrete mixture', 23, 'Table2，单位及11个配方')
    ym = excerpt('Yousri', 3, 'The AWA employed', 5, 'Materials，powder-based welan gum')
    yv = excerpt('Yousri', 3, 'One of a new generation', 6, 'Materials，VA为改性聚羧酸类')
    y_specs = [('M-0.5', .5, 2.1), ('M-0.75', .75, 2.1), ('M-1', 1, 2.1),
               ('M-0.5-10FA', .5, 2.5), ('M-0.5-15FA', .5, 2.5), ('M-0.75-15FA', .75, 2.5),
               ('M-1-15FA', 1, 2.5), ('M-10FA-VA', None, None),
               ('M-0.5-10FA-5SF', .5, 2.5), ('M-0.5-10SF', .5, 2.5), ('M-0.5-15SF', .5, 2.5)]
    y_recipes = [{'paper': 'Yousri', 'recipe_key': f'Yousri:Table2:{name}', 'specimen_id': name,
                  'awa_pct_cement': dose, 'sp_L_m3': sp} for name, dose, sp in y_specs]
    config['source_recipes'] += y_recipes
    y_filename = next(s['registry_pdf_filename'] for s in pilot['samples'] if s['paper'] == 'Yousri')
    for src in labels:
        if src['original_values']['PDF filename'] != y_filename:
            continue
        recipe = match_source_recipe(src['original_values']['Specimen ID'], y_recipes)
        if recipe['awa_pct_cement'] is None:
            add(src, 'Yousri', 'viscosity_modifier_pct_cement', .6, [yt, yv],
                'VA不是已确认welan-gum AWA；SP和AWA横线仍未知。', recipe=recipe)
        else:
            add(src, 'Yousri', 'awa_pct_cement', recipe['awa_pct_cement'], [yt], recipe=recipe)
            add(src, 'Yousri', 'sp_volume_L_m3', recipe['sp_L_m3'], [yt],
                '正文质量百分比与表中L/m³关系未解决，仅保留表2体积数值。', recipe=recipe)
            add(src, 'Yousri', 'antiwashout_product', 'powdered welan gum', [yt, ym], recipe=recipe)

    bt = excerpt('Brzozowski', 3, 'Table 1', 11, 'Table1，UCR1–3及各列单位')
    bm = excerpt('Brzozowski', 2, 'was 40% by mass.', 18, '§2.1，多糖AWA、UCR2的PP纤维、UCR3乳液')
    b_recipes = [{'paper': 'Brzozowski', 'recipe_key': f'Brzozowski:Table1:UCR{n}',
                  'specimen_id': f'UCR{n} (underwater)', 'awa_mass_kg_m3': awa, 'sp_L_m3': sp}
                 for n, awa, sp in [(1, 4.3, 12), (2, 4.3, 12), (3, 6.5, 6)]]
    config['source_recipes'] += b_recipes
    b_filename = next(s['registry_pdf_filename'] for s in pilot['samples'] if s['paper'] == 'Brzozowski')
    for src in labels:
        if src['original_values']['PDF filename'] != b_filename:
            continue
        recipe = match_source_recipe(src['original_values']['Specimen ID'], b_recipes)
        add(src, 'Brzozowski', 'awa_mass_kg_m3', recipe['awa_mass_kg_m3'], [bt], recipe=recipe)
        add(src, 'Brzozowski', 'sp_volume_L_m3', recipe['sp_L_m3'], [bt],
            '正文比重印刷单位异常及2%掺量与表格关系未解决，不做质量换算。', recipe=recipe)
        add(src, 'Brzozowski', 'antiwashout_product', 'polysaccharide-containing AWA', [bm, bt], recipe=recipe)
        if recipe['specimen_id'].startswith('UCR2'):
            add(src, 'Brzozowski', 'pp_mass_kg_m3', .9, [bt, bm], recipe=recipe)
            add(src, 'Brzozowski', 'fiber_type', 'fibrillated polypropylene', [bm], recipe=recipe)
        if recipe['specimen_id'].startswith('UCR3'):
            add(src, 'Brzozowski', 'copolymer_volume_L_m3', 5, [bt, bm],
                'styrene-acrylic产品体积，不是纯聚合物质量。', recipe=recipe)

    (ROOT / 'configs/material_feature_evidence_extended.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'annotations': len(config['annotations']), 'source_recipes': len(config['source_recipes']),
                      'records': len({a['source_row'] for a in config['annotations']})}, ensure_ascii=False))


if __name__ == '__main__':
    build()
