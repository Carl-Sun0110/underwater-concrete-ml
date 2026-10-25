"""生成未启用的训练配置和仅用来源分组构建的CV预案；绝不拟合模型。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from concrete_audit.baselines import digest, grouped_folds
from concrete_audit.scope_gate import MODES, load_training_gate, select_training_scope
from concrete_audit.train_compare import select_train_records


def run(folder):
    folder = Path(folder)
    labels_path = ROOT / 'outputs/provisional-labels-v1/exploratory_labels.json'
    split_path = ROOT / 'outputs/exploratory-baselines-v2/split_manifest.json'
    registry_path = folder / 'scope_registry.json'
    rows, groups, test_ids = select_train_records(json.loads(labels_path.read_text()), json.loads(split_path.read_text())['rows'])
    base_path = ROOT / 'configs/advanced_train_only.json'
    base = json.loads(base_path.read_text())
    plans = {}
    for mode in MODES:
        request = {'registry': str(registry_path.resolve()), 'registry_sha256': digest(registry_path), 'mode': mode}
        records, protected = load_training_gate(request, labels_path, split_path)
        selected, selected_groups, excluded = select_training_scope(rows, groups, records, mode)
        selected_groups = np.asarray(selected_groups)
        # X/y均为占位零，不访问真实配比值或强度；仅验证分组划分可执行性。
        dummy = np.zeros((len(selected), 1))
        dummy_y = np.zeros(len(selected))
        folds = grouped_folds(dummy, dummy_y, selected_groups, 5)
        cv = []
        for fold, (train, valid) in enumerate(folds, 1):
            inner = grouped_folds(dummy[train], dummy_y[train], selected_groups[train], 4)
            cv.append({'fold': fold, 'train_records': len(train), 'validation_records': len(valid),
                       'train_group_ids': sorted(set(selected_groups[train])),
                       'validation_group_ids': sorted(set(selected_groups[valid])),
                       'validation_sample_ids': [selected[i]['sample_id'] for i in valid],
                       'inner_folds': [{'fold': j, 'train_records': len(t), 'validation_records': len(v),
                                        'train_groups': sorted(set(selected_groups[train][t])),
                                        'validation_groups': sorted(set(selected_groups[train][v]))}
                                       for j, (t, v) in enumerate(inner, 1)]})
        config = {**base, 'experiment_id': 'scope-reviewed-' + mode,
                  'scope': '范围敏感性训练预案，尚未启用；不授予可信资格，不评估测试集',
                  'scope_gate': request, 'historical_scope_reproduction_only': False,
                  'training_execution_authorized': False,
                  'models': ['mean', 'ridge', 'svr', 'rf', 'xgb', 'mlp'], 'outer_folds': 5, 'inner_folds': 4}
        filename = 'planned_' + mode + '.json'
        (folder / filename).write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        plans[mode] = {'role': '方法证据范围内探索，不是最终可信模型' if mode == 'strict_evidence_scope' else
                              '仅隔离30条的混合未知范围敏感性分析，不能称严格主分析',
                       'train_records': len(selected), 'training_groups': len(set(selected_groups)),
                       'protected_test_records': len(test_ids), 'excluded_records': excluded,
                       'largest_group_fraction': max(list(selected_groups).count(g) for g in set(selected_groups)) / len(selected_groups),
                       'cv': cv, 'config_file': filename, 'training_executed': False,
                       'cv_uses_real_targets': False, 'same_validation_membership_for_all_models_in_this_cohort': True}
    result = {'plans': plans, 'models_trained': False, 'test_evaluated': False,
              'uncertainty': '严格范围仅5组，最大来源占81.25%；嵌套分组验证虽可执行，外推结论仍受来源数量限制。',
              'comparison_rule': '只在同一范围和同一组折内比较模型；不同范围或历史成绩不能直接比较高低来宣称改进。',
              'inputs_sha256': {str(labels_path): digest(labels_path), str(split_path): digest(split_path),
                                str(base_path): digest(base_path), str(registry_path): digest(registry_path)},
              'script_sha256': digest(__file__)}
    (folder / 'training_plan.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    summary = json.loads((folder / 'summary.json').read_text())
    views = json.loads((folder / 'scope_views.json').read_text())
    text = f'''# 水下浇筑范围门禁与后续训练预案

458条暂定标签记录中：212条有直接水下浇筑方法证据但仍待人工确认，216条范围未知，30条Assaad处理后重装模实验独立隔离。这里的范围判断不等于强度换算、绝对用量或整条可靠性已通过。

|视图|记录总数|原训练部分|来源组|原测试部分中范围符合数（仅审计）|
|---|---:|---:|---:|---:|
|严格方法证据范围|212|128|5|84|
|仅隔离已知30条（仍含未知范围）|428|269|19|145|

原145条测试ID完整保留，61条在严格范围内尚无方法证据；没有将它们移入训练，也没有测试预测或重新评价。14条核心输入未知记录仍保留在审计总体，不因门禁自动填补或投入训练。

## 范围证据与处理

Sonebi84条、Yousri11条、Brzozowski3条通过已编录完整配方ID与原文水下成型方法绑定；Zhang4条、Al-Selmi6条和Nasr104条沿用逐配方方法证据。Assaad30条的冲洗/渗水后重装模证据独立保存。不凭Underwater标签、题名、零值或算术一致认定可靠。

未知216条不会进入严格视图；在“仅隔离”视图中会保留，但必须明确称为混合未知范围的探索分析。两种视图均未改变原始文件、可靠状态或历史训练/测试划分。

## 交叉验证检查

严格视图保留旧折后的验证样本数为{views['strict_evidence_scope']['historical_outer_validation_counts']}，第5折为空，禁止直接沿用旧折。已仅用分组标签重建5外折×4内折的可执行预案；占位X/y均为0，没有用真实目标决定分组。

严格训练仅5个组，组大小为11、3、104、4、6，Nasr两篇保持同一防泄漏组；最大组104/128=81.25%。外层最少训练24条，内层最少训练13条。能运行不代表泛化结论稳定，应报告各来源误差和折间波动，不能只报最高R²。

仅隔离视图269条/19组单独建立相同规则的CV预案。不同视图样本不同，不能直接比较分数后宣称数据筛选带来提升；同一视图的各算法必须使用相同折。

## 后续训练预案（未启用）

保留均值、岭回归基线，比较SVR、随机森林、XGBoost、带Dropout的MLP。输入暂维持Water/Cem及计算比值两种表示，未知材料字段不填0加入模型；所有缩放和调参留在训练折内，沿用既有预定网格及MAE选择指标。

两个planned配置均为`training_execution_authorized=false`；实际训练入口会在拟合前拒绝执行。新训练必须提供范围档案路径、模式和版本指纹。旧比较配置缺少范围门禁会报错；如确需复现历史方案，必须在新副本显式设置`historical_scope_reproduction_only=true`，不能把复现当成新严格结果。旧基线入口同样仅保留显式历史复现用途。

历史测试集曾用于基线评价，本轮没有重评；未来严格模型仍需独立外部来源或前瞻数据验证，不通过重新随机划分来伪称全新未见测试。

## 输出及复现

- `浇筑范围审查与隔离清单.xlsx`：458条范围状态、30条隔离清单、216条未知范围。
- `scope_registry.json`：逐条证据档案，身份和上游文件版本绑定。
- `scope_views.json`：两种范围清单和原145条测试ID。
- `training_plan.json`：CV样本归属、组隔离及两份未启用的训练配置。

```bash
python scripts/prepare_scope_evidence.py
python scripts/run_scope_gate.py --output outputs/new-scope-gate
python scripts/plan_scope_training.py --folder outputs/new-scope-gate
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

新可信样本0；本轮没有训练。剩余问题是范围未知216条、严格训练来源仅5组，以及原有换算/掺量疑点尚未全部解决。
'''
    (folder / 'report.md').write_text(text)
    print(json.dumps({m: {'train_records': p['train_records'], 'training_groups': p['training_groups']}
                      for m, p in plans.items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    run(parser.parse_args().folder)
