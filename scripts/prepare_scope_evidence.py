"""将已读原文方法段与完整配方绑定；不把Underwater字段当成证据。"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.scope_gate import DIRECT, NON_DIRECT
from concrete_audit.material_features import match_source_recipe


def build():
    input_path = ROOT / 'configs/provisional_inputs.json'
    inputs = json.loads(input_path.read_text())
    material_path = ROOT / 'configs/material_feature_evidence_three_papers.json'
    material = json.loads(material_path.read_text())
    three_path = ROOT / 'configs/three_paper_field_evidence.json'
    priority_path = ROOT / 'configs/priority_paper_evidence.json'
    three = json.loads(three_path.read_text())
    priority = json.loads(priority_path.read_text())
    labels = json.loads((ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json').read_text())
    by_row = {r['source_row']: r for r in labels}
    claims = []

    def add(row, paper, state, refs, reason):
        raw = by_row[row]['original_values']
        claims.append({'source_row': row, 'specimen_id': raw['Specimen ID'],
                       'registry_pdf_filename': raw['PDF filename'], 'paper': paper,
                       'scope_state': state, 'evidence': refs, 'reason': reason})

    for sample in three['samples'] + priority['samples']:
        c = sample['claims']['CastingEnv']
        state = NON_DIRECT if sample['paper'] == 'Assaad2013' else DIRECT
        add(sample['source_row'], sample['paper'], state, c['evidence'],
            '冲洗/渗水后回收、重新装模，不属于直接水下浇筑实验，隔离但不删除。' if state == NON_DIRECT else
            '原文对应试验方法支持水下浇筑；其他字段及整条可靠性仍待核查。')

    def method_ref(key, page, anchor, count, location):
        pages = (ROOT / 'outputs/paper-evidence-01' / (key + '.txt')).read_text().split('\f')
        lines = pages[page - 1].splitlines()
        i = next(i for i, line in enumerate(lines) if anchor in line)
        return {'page': page, 'location': location, 'quote': '\n'.join(lines[i:i + count])}

    methods = {
        'Sonebi': method_ref('Sonebi', 3, 'The underwater strength was determined', 7, 'Mixing and test methods，水中圆柱与空气对照'),
        'Yousri': method_ref('Yousri', 3, 'Twelve', 15, 'Experimental programme，水下立方体与空气对照'),
        'Brzozowski': method_ref('Brzozowski', 2, 'The specimens for testing the compressive strength', 12, '§2.2.1，水下装模、切割立方体'),
    }
    selected = {}
    for a in material['annotations']:
        if a['paper'] in methods and a.get('source_recipe'):
            selected.setdefault(a['source_row'], a)
    for row, a in selected.items():
        recipe = match_source_recipe(a['specimen_id'], material['source_recipes'])
        if recipe['recipe_key'] != a['source_recipe'] or recipe['paper'] != a['paper']:
            raise ValueError('范围证据与来源配方绑定不一致')
        if by_row[row]['original_values']['CastingEnv'] != 'Underwater':
            raise ValueError('不能将空气分支纳入水下范围')
        add(row, a['paper'], DIRECT, [methods[a['paper']], *a['evidence']],
            '原文水下成型方法与已编录完整配方ID绑定；并非按论文题名或Underwater标签整批认定。')
    cfg = {'papers': material['papers'], 'claims': claims,
           'protected_inputs': [inputs[k] for k in ('raw', 'annotated', 'registry')] +
                               list(map(str, [input_path, material_path, three_path, priority_path])),
           'scope': '字段级实验范围证据，不等于整条可靠资格；未知拒绝进入严格范围视图'}
    (ROOT / 'configs/casting_scope_evidence.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'scope_claims': len(claims), 'direct': sum(c['scope_state'] == DIRECT for c in claims),
                      'isolate': sum(c['scope_state'] == NON_DIRECT for c in claims)}, ensure_ascii=False))


if __name__ == '__main__':
    build()
