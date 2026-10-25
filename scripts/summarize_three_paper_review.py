"""汇总本轮证据、范围疑点与实跑验证；不重新筛选或训练。"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import sys
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.material_features import append_excel_row
from concrete_audit.field_review import validate_evidence


def run():
    folder = ROOT / 'outputs/three-paper-review-v1'
    field_folder = ROOT / 'outputs/three-paper-field-check-v2'
    material_folder = ROOT / 'outputs/material-feature-three-papers-v1'
    config = json.loads((ROOT / 'configs/three_paper_field_evidence.json').read_text())
    checks = json.loads((field_folder / 'field_checks.json').read_text())
    summary = json.loads((field_folder / 'summary.json').read_text())
    material = json.loads((material_folder / 'summary.json').read_text())
    readiness = json.loads((material_folder / 'training_readiness.json').read_text())
    issues = json.loads((folder / 'source_issues.json').read_text())
    protocols = json.loads((folder / 'protocol_scope_proposals.json').read_text())
    pages = {key: (field_folder / (key + '.txt')).read_text().split('\f') for key in config['papers']}
    for issue in issues:
        validate_evidence(issue, pages[issue['paper']], config['papers'][issue['paper']]['sha256'])
    strengths = {r['source_row']: r for r in checks if r['field'] == 'CS_28d'}
    assert len(strengths) == len(protocols) == 134
    assert all(r['status'] == '已核实' for r in strengths.values())
    manifest = json.loads((ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json').read_text())
    split = {r['source_row']: r for r in manifest['rows']}
    assert all(split[row]['split'] == 'train' for row in strengths)
    nasr_groups = {split[r['source_row']]['leakage_group'] for r in protocols if r['paper'].startswith('Nasr')}
    assert len(nasr_groups) == 1
    # 对实际核查与材料模块保护的输入再次核对内容指纹。
    for path in [field_folder / 'manifest.json', material_folder / 'manifest.json']:
        protected = json.loads(path.read_text())['inputs_sha256']
        for source, expected in protected.items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
    gate = {'scope': '后续严格水下浇筑主分析的范围隔离建议，尚未改动任何既有样本或状态',
            'applies_to_future_runs_only': True, 'historical_split_unchanged': True,
            'reason': 'Assaad原文§3.4.3明确处理后回收、重新装模，不能等同于直接水下浇筑',
            'isolate_source_rows': [p['source_row'] for p in protocols if p['paper'] == 'Assaad2013'],
            'source_protocols': [p for p in protocols if p['paper'] == 'Assaad2013'],
            'new_trusted_samples': 0}
    (folder / 'strict_casting_scope_proposal.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2) + '\n')
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = '134条核查摘要'
    append_excel_row(sheet, ['原始行', '配方ID', '来源', '原库28d强度(MPa)', '原文对应强度(MPa)',
                             '数值辅助比较', '实验处理/浇筑方法', '范围处理建议', '人工状态', '新增可信资格'])
    for p in protocols:
        s = strengths[p['source_row']]
        append_excel_row(sheet, [p['source_row'], s['specimen_id'], p['paper'], s['excel_value'], s['paper_value'],
                                 '数值一致，仅字段级辅助结论', p['protocol'], p['scope_action_proposal'], '待核查', '否'])
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    sheet = book.create_sheet('关键问题')
    append_excel_row(sheet, ['论文', '问题', '涉及记录数', '处理'])
    for issue in issues:
        append_excel_row(sheet, [issue['paper'], issue['issue'], len(issue['affected_source_rows']), issue['action']])
    book.save(folder / '三篇原文核查摘要.xlsx')
    log = Path('/tmp/underwater-three-paper-tests.log').read_text()
    match = re.search(r'Ran (\d+) tests.*?\n\nOK', log, re.S)
    assert match
    (folder / 'tests.log').write_text(log)
    validation = {'tests_passed': int(match.group(1)), 'failures': 0, 'errors': 0,
                  'field_run': summary, 'material_run': material, 'source_issues_evidence_validated': True,
                  'original_strength_numeric_matches': len(strengths),
                  'all_134_in_fixed_training_split': True, 'nasr_papers_kept_in_same_leakage_group': True,
                  'input_hashes_rechecked': True, 'strict_scope_isolation_proposed': len(gate['isolate_source_rows']),
                  'model_trained': False, 'test_performance_evaluated': False}
    (folder / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    text = f'''# 新上传三篇原文核查报告

核查134条：Nasr静水40条、Nasr流动水64条、Assaad30条，均在既有训练部分。按来源定向核查，不据此估计全库通过率。

134条原库CS_28d均与对应原表的实测强度数字一致。Nasr明确选择28d FW/SW列；Assaad按原库CastMethod对应冲洗/渗水分支，未用目标值反推列位。没有用空气强度、相对强度或回归式替代目标。

**数字匹配不是整条可靠性批准。新增“已核实可用”样本仍为0。**

## 必须保留的问题

- Assaad30条：§3.4.3为冲洗或渗水后回收、重新装入100×200mm圆柱，再养护至28天。与严格水下浇筑范围不同，建议后续主分析隔离；不删除、不覆盖当前数据，也未更改既有划分或可靠状态。
- 两篇Nasr的UPA加在外部水环境，不是混凝土内掺AWA。按原表保存水体浓度%，质量/体积定义仍未知，不换算成AWA kg/m³。
- 两篇材料段均出现0.05%，C2配比表及配比段给0.005%；涉及38条C2记录。分开保留矛盾，不擅自宣布哪个数字正确。
- Nasr静水Table1中SP单位L/100kg水泥，正文§3.4写kg/100kg水泥。40条质量SP继续待核；不通过猜测产品密度去证明原库kg/m³。
- Assaad缓凝剂为UWC水泥质量的0.6%，CEM砂浆为0.4%；30条独立质量候选计算均保留参数来源，未回写原库0。
- Assaad AWA/HRWR按原文“% of cement”记录；原表横线不转成零，UWC与CEM列不混用。
- 静水表2印刷配方5M2C1/6M3C1按已审查的明确别名映射到M2C1/M3C1，保留印刷ID；没有普遍删除数字或模糊匹配。
- 用户允许的0.95、0.8、0.76仍只是暂定换算规则；本轮未把实测值匹配当作换算系数或“无密度假设”的证明。

## 实际运行结果

字段核查1790项，其中1284项有定位证据，506项本轮未编录。辅助状态：956数值/规格等一致、30需转换、298待人工确认、506证据不足；所有记录均需人工最终确认。未编录不表示原文没有报告。

材料候选层累计{material['records_with_evidence_candidates']}条有至少一项定位声明，共{material['field_annotations']}项声明（此前108条/447项）；这些不是完整可用样本数。当前训练部分至少一项候选覆盖158/299条，17篇未匹配训练原文减少至{len(readiness['unmatched_source_pdfs'])}篇。没有一项新增材料字段完整覆盖299条，不能整体补0入模。

完整{match.group(1)}项测试通过。核查配置、原始Excel、批注版、可靠状态档案、暂定标签、划分和论文输入均通过内容指纹复核。两篇Nasr仍在同一防泄漏组。本轮不训练、不重新评价测试集。

## 输出及复现

- `outputs/three-paper-review-v1/三篇原文核查摘要.xlsx`：134条简表和关键问题。
- `outputs/three-paper-field-check-v2/小批字段核查.xlsx`：1790项详细字段核查及证据。
- `outputs/material-feature-three-papers-v1/材料字段与证据.xlsx`：458条独立候选视图及累计证据。
- `source_issues.json`、`protocol_scope_proposals.json`、`strict_casting_scope_proposal.json`：疑点和30条范围隔离建议，均未变更原档案。
- `validation.json`、`tests.log`：实跑结果。

```bash
python scripts/prepare_three_paper_review.py
PYTHONPATH=src python -c 'from concrete_audit.field_review import run; run("configs/three_paper_field_evidence.json", "outputs/new-three-paper-check")'
python scripts/run_material_features.py --config configs/material_feature_evidence_three_papers.json --output outputs/new-three-paper-materials
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

编录脚本使用本轮保存的PDF提取文本；执行核查时会重新读取原PDF并校验引文与版本，文本匹配不自动证明人工解释正确。

## 本轮修改

新增`src/concrete_audit/source_tables.py`、`tests/test_source_tables.py`、`scripts/prepare_three_paper_review.py`、`scripts/summarize_three_paper_review.py`及两份新证据配置。扩展`material_features.py`、`test_material_features.py`，更新README/CHANGELOG。原文件及历史输出均保留。

下一步优先落实严格水下浇筑范围门禁和实验条件字段，再明确训练内敏感性方案。当前不急于追求更高R²，也不要求一次补齐剩余论文。
'''
    (folder / 'report.md').write_text(text)
    print(json.dumps({'original_strength_matches': 134, 'scope_isolation_proposed': 30,
                      'tests_passed': int(match.group(1)), 'new_trusted_samples': 0}, ensure_ascii=False))


if __name__ == '__main__':
    run()
