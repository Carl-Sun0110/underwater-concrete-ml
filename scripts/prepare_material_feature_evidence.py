"""把已阅读的两篇原文编录为独立语义声明；不外推到未指定配方。"""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.material_features import FEATURES

source = json.loads((ROOT / 'configs/priority_paper_evidence.json').read_text())
folder = ROOT / 'outputs/priority-paper-review-v1'
pages = {key: (folder / (key + '.txt')).read_text().split('\f') for key in source['papers']}


def ref(paper, page, anchor, after, location):
    """按已确认位置保存逐字引文，运行阶段将对实际PDF重新校验。"""
    lines = pages[paper][page - 1].splitlines()
    i = next(i for i, line in enumerate(lines) if anchor in line)
    return {'page': page, 'quote': '\n'.join(lines[i:i + after]), 'location': location}


annotations = []
for sample in source['samples']:
    paper = sample['paper']
    claims = sample['claims']

    def add(feature, value, refs, note='', **kwargs):
        unit, basis = FEATURES[feature]
        annotations.append({k: sample[k] for k in ('source_row', 'specimen_id', 'registry_pdf_filename', 'paper')} | {
            'feature': feature, 'value': value, 'unit': unit, 'basis': basis,
            'evidence': refs, 'method': 'curated_source_reading', 'note': note,
            'human_confirmation': '待核查', 'admitted_to_model': False, **kwargs})

    if paper == 'Zhang2025':
        system = ref(paper, 5, 'To investigate the effects of polymer', 3, '§2.2 本研究四组UN-UHPC')
        material = ref(paper, 4, 'achieve excellent strength', 7, '§2.1 钢纤维和UWB-II产品组成')
        add('material_system', 'UN-UHPC', [system])
        add('antiwashout_product', 'UWB-II polymer blend', [material], '混合产品，不等同于纯聚合物或其他AWA产品。')
        add('fiber_type', 'copper-coated steel', [material])
        dose = claims['AWA_Dos']['paper_value']
        extra = {'explicit_absence_quote': 'had no polymer addition'} if dose == 0 else {}
        add('polymer_pct_cement', dose, claims['AWA_Dos']['evidence'],
            '比例0/.015/.025/.030分别为0/1.5/2.5/3%；不使用未核实的kg/m³水泥基准。', **extra)
        add('steel_mass_ratio_cement', .16, claims['StF']['evidence'], 'kg钢纤维/kg水泥，不是体积分数。')
    else:
        mixing = claims['CastingEnv']['evidence']
        product = ref(paper, 5, 'pan-type mixer', 6, '§2.2 welan-gum powder')
        density = ref(paper, 5, 'Table 3. Polyvinyl', 11, 'Table3 PVA密度1.28g/cm³')
        add('material_system', 'ECCUWC', mixing)
        add('antiwashout_product', 'welan-gum powder', [product])
        add('fiber_type', 'PVA', [density])
        add('awa_pct_binder', claims['AWA_Dos']['paper_value'], claims['AWA_Dos']['evidence'],
            '胶凝材料质量百分比；不与Zhang水泥基准聚合物列合并。')
        add('awa_mass_kg_m3', claims['AWA']['paper_value'], claims['AWA']['evidence'],
            '原表4直接报告，胶材1502/1503矛盾另存，不从百分比重算。')
        vol = 1.5 if sample['source_row'] <= 1079 else 1.75
        add('pva_vol_pct', vol, mixing + claims['SyF']['evidence'], '命名规则和表4共同定位配方。')
        add('pva_density_kg_m3', 1280, [density], '原文PVA密度1.28g/cm³×1000。')
        add('pva_mass_kg_m3', claims['SyF']['paper_value'], [density] + mixing + claims['SyF']['evidence'],
            '原文表4直接质量值，并以原文体积分数和PVA密度独立核对。',
            method='pva_volume_density', parameters={'vol_pct': vol, 'density_kg_m3': 1280})

config = {'papers': source['papers'],
          'protected_inputs': [source[k] for k in ('raw', 'annotated', 'registry')],
          'scope': '两篇已阅读论文10配方，458条探索记录的独立候选层，未知不填0，禁止自动入模',
          'annotations': annotations}
(ROOT / 'configs/material_feature_evidence.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
print(f'已编录{len(annotations)}项材料候选声明；仍需人工确认。')
