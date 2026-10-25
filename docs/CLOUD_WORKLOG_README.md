# 水下混凝土数据库与探索性机器学习

2026-10-24明确协议的Baseline V1已独立实现：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_baseline_v1.py`。直接重读Total Excel，使用存储Converted标签，主集为有直接水下成型范围证据的212条，扩展探索集467条。固定参数RF/XGBoost/CatBoost，比较隔离近重复配比的随机划分、论文组留出和五折分组验证。完整报告、筛选记录、图、代码副本与日志位于`outputs/baseline_v1`；不与历史不同标签/范围/划分的分数混为同一实验。

Baseline V1.1：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_baseline_v1_1.py`，锁定V1论文留出，仅训练集内比较两/十一特征和中位数/KNN插补。报告位于`outputs/baseline_v1_1/report.md`，保存仅训练集拟合的候选模型。`scripts/predict_strength.py --model ... --input ... --output ...`提供CSV预测，标记缺失与超训练范围，额外强度列不作为特征。当前预测仍为探索演示，不等于可靠科研或工程结论。

小批文献线索与水泥零值假设：`scripts/run_priority_evidence_zero.py`生成4篇/26条的原表单元格核查包，并仅在既定扩展训练集比较Zhang 2023四条Cem0的语义假设。报告位于`outputs/priority-evidence-zero-v1/report.md`。当前网络代理拒绝请求，未取得新原文；原表提示、题名和模型分数均不作为PDF已核实证据。已发布模型及原始标签不被假设实验覆盖。

阶段报告与程序交付：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/build_stage_report.py`，依赖`requirements-report.txt`。输出`outputs/stage-report-v1`内的Word/Markdown教师报告、来源配对重采样诊断及预测ZIP。报告不新增训练或留出评价；重采样分位范围仅描述来源构成稳定性。ZIP逐文件哈希已验证，实际解压到独立临时目录运行成功。默认输出目录须不存在。

**当前工作模式：文献核查与机器学习并行推进。** 输入缺失或尚未逐项核查不再自动阻止探索训练；标签必须有原表来源，原始文件只读，探索结果与正式科研结论分开。见[双轨方案](docs/parallel_research_plan.md)。

新增探索入口：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_parallel_exploration.py --output outputs/parallel-exploration-v1`。输出包括分层快照、字段可用性、来源分组五折结果、逐样本折外预测、误差核查优先列表和演示模型。默认28个固定参数对照，145条历史留出本轮不参与训练、插补、选模或评估。

第二轮入口：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_nested_exploration.py --output outputs/parallel-nested-v2`，依赖`requirements-nested.txt`。复用相同AB开发记录及外层折，在训练来源内部三折调参，对照RF、XGBoost、CatBoost和配比比值特征。比值仅在原表单位可比假定下作探索解释；不补抗压标签，不将未知掺合料当0求胶凝总量。输出包含全部内层搜索记录和同队列基线比较。初次nested-v1因CatBoost/sklearn标签兼容失败，保留日志；v2包含适配与完整重跑。

第三轮补齐SVR/MLP：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_svr_mlp.py`；恒定列诊断对照为`scripts/run_scaled_comparison.py`；两篇原PDF配比复查为`scripts/recheck_two_papers.py`。CPU依赖见`requirements-complete-models.txt`。输入与目标缩放均放在训练折内，MLP带Dropout。原始大误差与负预测保留，恒定列移除只按训练折拟合。汇总见`outputs/complete-algorithm-comparison-v1/report.md`及Excel；历史留出本轮继续不使用。每个脚本默认输出目录须不存在。

第四轮来源选参/证据版本对照：`PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python scripts/run_source_weighting.py`。同一283条开发记录、同一外层折、同一参数范围，对照旧折平均选参与来源组等权选参，以及原字段/单位敏感性×有无文献配比特征。没有给训练样本加权；单位处理仅生成独立未知值矩阵，原始数值不回写。结果和逐格依据见`outputs/source-balanced-evidence-v1/analysis/report.md`及Excel。

以下是早期只读审计阶段的历史说明，不代表当前模型开发的前置条件。

本阶段只检查文件、生成审计档案、管理可靠状态；没有预处理、模型训练或目标驱动的样本删除。
原始文件保留在 `/workspace/attachments`，不复制覆盖、不保存回写。仓库起始为空，无原有代码可复用。

## 运行

Python 3.12，直接依赖为 `requirements.txt` 中的 openpyxl 3.1.5；测试使用标准库 unittest。
当前环境已有依赖，无须安装。新环境可在虚拟环境中运行 `python -m pip install -r requirements.txt`。
在本仓库目录执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python run_audit.py --config configs/inputs.json --output outputs/my-new-run
```

输出目录必须不存在；重复运行请指定新目录，程序拒绝覆盖已有报告。配置明确指定两份附件路径，不按名字猜测原始版本。
路径发生变化时编辑配置，不能用同名但不同内容的文件冒充旧版本。每次记录输入哈希、代码指纹、环境及参数。
当前报告见 `outputs/final-audit/report.md`。

## 目录与关键函数

- `src/concrete_audit/audit.py`：工作簿结构、参数字典、原始记录、合并身份、批注及格式线索、版本差异与完整性统计。
- `src/concrete_audit/reliability.py`：有限数值解析、换算机械检查、人工决策门槛。
- `configs/inputs.json`：明确输入位置；`evidence` 为实际证据文件列表，`decisions` 默认空。
- `tests/test_audit.py`：算术不授信、缺证据拦截、版本冲突、输入不变及输出不覆盖等测试。
- `outputs/<run>/registry.json`：本阶段审计主档，不是模型输入表。

`identity_values()` 只展开原本存在的A/B列合并单元格，保留锚点；不会把普通空白默认为上一来源。
`comment_rows()` 仅解析明确的单元格范围。关联可能有遗漏或过宽，应结合 `comments.json` 人工确认。
`conversion_check()` 仅检查强度数值、系数、试件字段和换算算术；不会验证试件默认规则或系数的文献依据。
`run()` 将来源哈希+工作表+行号作为记录ID，先核对PDF、Research Group、Specimen ID，再关联历史记录。
不同版本的差异单独报告，不将批注版等同于《人工核查版》，也不回写旧表。

## 可靠状态规则

1. **待核查**：默认状态；机械检查通过、未触发旧规则、批注声称已核对，均不能自动升级。
2. **已核实可用**：要求显式人工决策绑定记录ID和源文件哈希，填写核查人、日期、理由、证据文件哈希及页码/表格位置；来源、强度依据、密度假设、单位、零值语义五项均已核实，明确未使用假设密度反算，显式无未解决问题，并通过水下与机械条件。
3. **已排除**：本次自动排除仅限原表明确为Air的记录，理由是研究范围；其他排除需显式人工决策。旧筛选表“排除”保留在 `historical_status`，不冒充当前最终结论。

未知状态保存为“未知”，不从 Density 正值或零值推断有无假设。批注全文属于附件陈述，不是程序独立查阅论文后的结论。
人工决策格式为JSON数组；初次无决策时保持配置 `decisions: null`。每项字段包括：
`record_id, source_sha256, status, reviewer, reviewed_at, reason, evidence, checks, density_assumption, unresolved_reasons`。
`evidence` 每项为 `{"sha256":"实际证据哈希","location":"页码/表格"}`，证据文件也必须在配置 `evidence` 中提供。
`checks` 包含 `source, strength_basis, density_assumption, feature_units, zero_semantics`；可用资格要求各项为“已核实”。
`density_assumption` 要求“已核实未使用假设密度反算”，`unresolved_reasons` 要求 `[]`。
这些是可追踪的人工声明；程序仅验证声明完整性及证据文件存在，不能自动证明证据支持结论。
每次人工决策记录在 `manual_review_events.json` 和逐条处理历史中，不删除原始核查线索。

## 当前文件缺口及边界

已有两份附件：筛选方案与核查清单工作簿、批注核对版(1)工作簿；原始仓库无Excel和论文。
缺少同名的《水下混凝土_人工核查版.xlsx》《水下混凝土_筛选结果_20261012.xlsx》、来源论文PDF、完整换算依据和逐条人工核查档案。
批注版的54个来源文件名逐项列在 `evidence_inventory.json`；Paper List中的条目不等于本机已有论文。
未新增任何可信样本，不推断历史版本的差异原因，不处理零值，不补齐强度，不按异常值删样本。
完成模块1、2后暂停，等待审阅；后续特征分析和训练需要另行确认。

## 最新盘点（按新任务文本完成第一、二步）

当前完整材料清单及处理历史见 `outputs/stage12-current-inventory/report.md`，取代上文早期的文件缺口描述。已收到4份数据Excel、5篇原始论文和1份说明报告；并非仍缺全部论文。

```bash
python scripts/inventory_stage12.py --output outputs/new-inventory-run
```

本阶段画像仅区分物理空白、占位、零与公式，不自动判真；后续字段级Agent与人工核查表尚待实现。变更见 `CHANGELOG.md`。

## 小批字段核查试点

本地试点使用 `configs/pilot_evidence.json` 中助手阅读原文后编录的证据，逐项检查PDF指纹、页码、摘录、样本身份、单位和值。不是外部LLM自动提取，不替代人工确认。

```bash
python scripts/run_pilot_review.py --output outputs/new-field-review
PYTHONPATH=src python -m unittest discover -s tests -v
```

12条定向样本涉及5篇论文、165个字段；123项带定位证据，42项尚未编录。现有结果见 `outputs/pilot-field-review-v2/report.md`。
六种字段核查状态与三种样本可靠状态分开，字段“已核实”仅为辅助结论，所有条目仍需人工复核。程序不修改原值和可靠状态主档。缺密度不计算质量；可确认参数的计算结果另存为“候选计算，未审批”。

`prepare_pilot_evidence.py` 是本次证据编录的重建脚本，依赖已有提取文本；它不是全文自动核查器。扩展样本前应先读取原文、记录明确的字段级证据，再更新配置，不能仅复制同篇结论覆盖所有记录。

## 用户暂定系数下的独立探索标签

```bash
python scripts/run_provisional_labels.py --output outputs/new-provisional-run
python scripts/profile_provisional_features.py --source outputs/new-provisional-run/exploratory_labels.json --output outputs/new-provisional-run/feature_readiness.json
```

规则见 `configs/provisional_strength_policy.json`，输入见 `configs/provisional_inputs.json`。
使用每条记录既有0.95、0.8、0.76，不猜测缺失、不替换其他系数。原527候选中458条符合此范围，69条原样保留隔离；无新增可信资格。
`Provisional_Converted_CS_28d` 独立保存未取整结果，原始强度和原换算值不变。
输出含完整审计字段，不是模型输入矩阵；不得将目标或换算/质量字段直接作为特征。
结果见 `outputs/provisional-labels-v1/report.md`。当前23项测试通过，未训练模型。

## 特征与零值策略比较

```bash
python scripts/run_feature_plan.py --output outputs/new-feature-plan
```

配置 `configs/feature_policy.json` 规定核心、扩展及暂缓特征，禁止目标、强度派生、身份及质量状态作为输入。
所有零值仅在独立诊断视图标为未知，原始值保持不变。458条中444条满足Water/Cem正值；其他条件及未知比例的样本保留敏感性见 `outputs/feature-plan-v1/report.md`。
不选定缺失阈值、不拟合填补或缩放、不训练；正数不代表已核实。当前28项测试通过。

## 探索性两输入基线（已运行）

用户在基线方案后要求继续，当前执行授权保存在 `configs/baseline_experiment.json`；上游标签文件中的历史training_authorized=false保持不变，不代表本次授权修改了历史核查状态。

```bash
python scripts/run_baselines.py --output outputs/new-baseline-run
```

依赖版本见 `requirements-modeling.txt`。结果见 `outputs/exploratory-baselines-v2/report.md`：444条，299训练/145测试，来源组完全隔离；训练部分执行5外折×4内折嵌套分组验证，最终训练部分5折调参。
仅Water/Cem输入，训练均值和岭回归均表现弱，岭回归测试R²=-0.0139、MAE=9.2249 MPa。未通过更换种子或使用测试调参提高分数。
原始数据和可靠状态未改变，不作为正式可信模型；模型对象和预测文件均保留。当前34项测试通过。

## 仅训练部分的非线性模型比较

```bash
python scripts/run_train_comparison.py --output outputs/new-train-comparison
```

使用原299条训练样本和相同5外折×4内折分组CV；145条测试集本轮未重新预测/评估。
输入为Water/Cem，以及新增计算比值Water/Cem的对照。未将未确认材料或未知零值直接投入训练。
比较均值、岭回归、RBF-SVR、随机森林；结果见 `outputs/train-only-comparison-v1/report.md`，均为训练内CV，不能当测试分数。
8组配置完整保留；RF+计算比值的折外MAE约7.768 MPa、R²约0.161，改善有限。未进行XGBoost/MLP或最终模型拟合。
当前38项测试通过，原始数据、可靠状态与固定划分保持不变。

## XGBoost与带Dropout的MLP（已运行，测试集未重评）

```bash
bash scripts/setup_advanced_env.sh
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
/workspace/.venv-underwater-ml/bin/python scripts/run_train_comparison.py --config configs/advanced_train_only.json --output outputs/new-advanced-run
/workspace/.venv-underwater-ml/bin/python scripts/summarize_four_models.py --advanced outputs/new-advanced-run --output outputs/new-four-model-summary
```

CPU依赖安装在 `/workspace/.venv-underwater-ml`，安装版本和索引见requirements-advanced.txt、requirements-torch-cpu.txt。
MLP使用PyTorch两层32/16隐层及实际Dropout，输入/目标缩放均在训练折内，固定200轮，无外层早停。
XGBoost与MLP沿用原299条、5外折×4内折，只做训练CV；145条测试未重新评估。
四算法汇总见 `outputs/four-model-comparison-v1/report.md`。比值表示下XGBoost MAE约6.251 MPa、R²约0.216；MLP MAE约6.750但R²为负，保留弱结果。没有生成最终可信或部署模型。
43项测试在独立环境全部通过，无跳过；未知材料和原始标签核查状态未改变。

## 训练折外误差诊断（不重训、不动测试）

```bash
python scripts/run_error_analysis.py --output outputs/new-error-analysis
python scripts/summarize_error_analysis.py --folder outputs/new-error-analysis
```

结果见 `outputs/training-error-analysis-v2/report.md`：Zhang2025四条贡献XGBoost约49.7%平方误差；Al-Selmi2026六条贡献MLP约44.7%。21条外折输入超范围；55组相同Water/Cem却不同目标涉及275条记录。
这些是诊断线索，不能据此删记录、判定数据错误或按误差改变研究范围。新两篇全文未提供，题录仅作为线索。
47项测试在独立CPU虚拟环境中全部通过；原始数据、可靠状态和145条测试保持不变。

## 两篇重点误差来源原文核查（新上传后）

已经读取Zhang及Al-Selmi两篇全文，检查原表942–945、1077–1082共10条。报告见 `outputs/priority-paper-review-v1/report.md`，字段表见 `outputs/priority-paper-field-check-v2/小批字段核查.xlsx`。
原文支持Zhang四条28d原始强度（图上明确数字，非估读），但其Table3为归一化比例，现有kg/m³基准仍未知。Al-Selmi的Water605/Cem1202等确为表4所报kg/m³，水下浇筑和100×200mm圆柱也有据；五个28d强度图无明确数值标签，精度尚待确认。
原文内部表述矛盾另存，不把论文回归方程替代实测标签；体系建议标签不修改原始/可信档案。
新增图像证据的PDF/图像内容哈希绑定验证，完整50项测试通过。没有重训或重评测试集。

本轮配置为 `configs/priority_paper_evidence.json`。可在Python环境中调用 `concrete_audit.field_review.run(config_path, new_output_path)` 重新生成字段核查表，新输出目录必须不存在。证据编录来源重建脚本为 `scripts/prepare_priority_paper_review.py`，依赖本次实际提取文本/页图；它不是自动阅读全文判真工具。
新上传后实际可用原始论文已增至7篇，另有1份数据库说明PDF；此前报告的“5篇”是历史时点。

## 材料语义候选字段（不自动入模）

```bash
python scripts/prepare_material_feature_evidence.py
python scripts/run_material_features.py --output outputs/new-material-feature-mapping
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

已运行结果见 `outputs/material-feature-mapping-v1/report.md`、`材料候选字段.csv` 和 `evidence_snapshot.json`。
范围为458条探索记录；仅为本轮两篇原文10配方编录68项声明，其余值保持未知。这不是对其他论文证据是否存在的判断，也不是最终可信数据集。
水泥质量百分比、胶材质量百分比、PVA体积分数及kg/m³各自保存；SX1明确未掺聚合物的0不会扩散到其他配方。
`validate_annotation`检查字段白名单、单位基准与PDF引文；`pva_mass`使用原文纤维密度核对体积分数换算；`make_views`保留未知和原始字段；`run`重读PDF并核对输入哈希，只写新目录。
材料体系和产品名称是待人工确认的候选描述，不能直接按论文身份编码入模。输出不包含目标值、强度派生量或模型预测；测试部分仅统计非目标覆盖，不重新评分。
完整55项测试通过；原始数据、可靠状态档案、模型及既有划分均未更改。

## 既有论文材料字段扩展与训练来源缺口

```bash
python scripts/prepare_existing_material_evidence.py
python scripts/run_material_features.py --config configs/material_feature_evidence_extended.json --output outputs/new-expanded-materials
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

本轮完成输出是 `outputs/material-feature-expansion-v2`。扩展后108条记录、447项字段声明、17类语义字段；108条中24条在训练集、84条在测试集，均仅做输入审计，不改变划分。108不是新增可信样本数量。
`材料字段与证据.xlsx`含458条候选视图、447项逐条证据、字段规则及17篇未匹配训练来源全文的清单；`training_readiness.json`只按299条训练记录统计缺口，不按目标或模型误差排序。
Sonebi的84条按表3/4中完整配方ID绑定，液态纤维素剂量与welan gum质量百分比区分；Yousri的11配方中VA独立保存；Brzozowski的PP纤维及SP/乳液体积按表1保存。没有恢复Wang2021和Chaikasatsin的非当前探索记录。
`match_source_recipe`拒绝近似/重复配方匹配；`training_readiness`检查训练覆盖和来源缺口；Excel导出仅在显示层去除XML禁用控制字符，原文JSON和引文校验不变，字符串按文本保存。
完整60项测试通过，原始文件和可靠状态指纹未变。未重训、未重新评价测试集；所有候选字段仍待人工核查。
首次Excel导出因原PDF控制字符失败的目录 `material-feature-expansion-v1` 已标记为不完整，不能使用；v2为修复后完整输出。

## 新上传Nasr两篇与Assaad原文核查

```bash
python scripts/prepare_three_paper_review.py
PYTHONPATH=src python -c 'from concrete_audit.field_review import run; run("configs/three_paper_field_evidence.json", "outputs/new-three-paper-check")'
python scripts/run_material_features.py --config configs/material_feature_evidence_three_papers.json --output outputs/new-three-paper-materials
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

实际报告见 `outputs/three-paper-review-v1/report.md`，简表为同目录`三篇原文核查摘要.xlsx`；详细核查为`outputs/three-paper-field-check-v2`，更新候选层为`outputs/material-feature-three-papers-v1`。摘要生成脚本`summarize_three_paper_review.py`使用这些已运行目录，并重新核对输入指纹与疑点引文。
134条对应原始28d强度数值一致，不代表整条可靠。Assaad30条是暴露后重新装模，已出严格水下浇筑范围隔离建议，未改既有档案或历史数据。Nasr UPA是外部水体浓度，C2正文/表格有0.05/0.005%矛盾，静水SP表格/正文存在体积/质量单位矛盾。
`source_tables.py`显式绑定表格列、龄期、FW/SW及冲洗/渗水程序，禁止根据目标值反推分支，拒绝缺列、重复和模糊配方。候选层累计242条/1245项声明，仍待人工核查；训练部分至少一项候选覆盖158/299条，不以未知补0。
完整65项测试通过；新可信样本0，原始及可靠状态不变，未重训或重评测试集。两篇Nasr继续处于同一防泄漏组。

## 严格水下浇筑范围门禁与训练预案

```bash
python scripts/prepare_scope_evidence.py
python scripts/run_scope_gate.py --output outputs/new-scope-gate
python scripts/plan_scope_training.py --folder outputs/new-scope-gate
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

已运行结果见 `outputs/casting-scope-gate-v1/report.md` 和 `浇筑范围审查与隔离清单.xlsx`。458条分为212条有水下浇筑方法证据但待人工确认、216条范围未知、30条非直接浇筑实验隔离；原可靠状态均不改变。
严格方法证据范围在既有训练部分剩128条/5组，测试范围审计84条；仅隔离30条的探索视图为269训练条/19组，未知范围仍保留。原145条测试ID完整保留，没有移入训练或重新评价。严格视图旧CV第5折为空，已仅用分组标签重建5外折×4内折预案；未使用真实目标或拟合模型。
`build_records`验证配方/论文及PDF摘录；`scope_allows`让未知状态无法通过严格门禁；`select_training_scope`只过滤固定训练部分，不读取目标；`load_training_gate`把审查结果绑定原文、标签和划分版本。

**此前README中的旧训练命令现在默认会停止。** 新比较训练必须配置`scope_gate`的档案路径、指纹和模式；两份`planned_*.json`仍为`training_execution_authorized=false`，只供审查。确需历史复现时，应创建新配置副本并显式设置`historical_scope_reproduction_only=true`，不能将历史范围结果当作新严格结果；旧基线入口同样仅接受显式历史复现。
严格训练5组中最大组104/128=81.25%，外层最少训练24条、内层最少13条；预案可执行不代表独立泛化结论稳定。所有新材料及可靠性疑点保持未决，新增可信资格0。
完整72项测试通过，并实际验证未启用预案及缺少范围配置均在输出/模型拟合前停止；本轮没有重训或测试评分。

## 范围门禁后的训练内探索（用户继续后已执行）

用户继续授权后，新建`configs/scope_strict_train_only_v1.json`和`configs/scope_isolation_sensitivity_v1.json`，仅在新副本启用执行，保留原未启用预案及其指纹。模型、输入表示、网格、种子和CV折数均与预案一致。

```bash
/workspace/.venv-underwater-ml/bin/python scripts/run_train_comparison.py --config configs/scope_strict_train_only_v1.json --output outputs/new-strict-training
/workspace/.venv-underwater-ml/bin/python scripts/run_train_comparison.py --config configs/scope_isolation_sensitivity_v1.json --output outputs/new-isolation-training
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

已完成原始训练结果分别在`outputs/scope-strict-train-only-v1`和`outputs/scope-isolation-sensitivity-v1`；汇总报告及Excel在`outputs/scope-training-comparison-v2`。`summarize_scope_training.py`使用这两个实际结果目录进行只读审计与汇总。
共24项实验、120个外折、4764条折外预测。严格范围128条/5组，MAE最小为XGBoost核心两输入，MAE9.915MPa、R²0.0573、来源组等权MAE20.644MPa；表现仍弱。仅隔离敏感性范围269条/19组，MAE最小为MLP加计算比值，MAE6.295MPa、R²0.2137；不同范围分数不能直接比较。
严格范围MLP加比值在Al-Selmi组出现约−133MPa预测，结果完整保留，未截断或临时改网格。逐来源误差和全部超参数记录均可在Excel查看。
新增`scope_results.py`与3项测试检查相同外折、训练身份覆盖、测试隔离及来源等权汇总。完整75项测试通过；实际折与事前预案一致，原始及可靠状态指纹不变，145条测试本轮未预测/评分。没有选定或部署最终可信模型。

## 原文细骨料/水泥比与折内预处理检查

已为严格训练128条/5组编录新的细骨料/水泥质量比候选。Zhang直接使用原文归一化Sand/Cement=1.05；其他来源用同表细骨料和水泥kg/m³相除，未用密度反推绝对用量。分母为水泥，不是胶材；Yousri的F.A.细骨料与FA粉煤灰分开，Brzozowski的质量列与体积列分开。

```bash
# 首次编录生成新配置；配置已存在时不要重复执行此行
python scripts/prepare_fine_aggregate_ratio.py
/workspace/.venv-underwater-ml/bin/python scripts/run_evidence_feature_plan.py --output outputs/new-evidence-feature-plan
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

实际输出见`outputs/evidence-feature-plan-v1/report.md`和`新增物理特征与预处理核查.xlsx`，包含128条X候选、原文参数和5折预处理记录。本轮真正执行5次外训练折缩放，验证折只transform，不用y、不填补、不拟合预测模型，测试行完全未用于预处理。未来调参仍需把缩放放回每个内训练折拟合。
水环境/水流/UPA等虽有64或104条值，均只涉及Nasr一组；该组被留出时其余训练折无此列证据，未直接入模。新质量比有完整覆盖但仍待人工确认，也不证明原Water/Cem或暂定目标已经可信。
`evidence_features.py`实现原表列提取、同单位质量比、固定三列输入变换和来源折覆盖诊断。新增5项测试，完整80项通过，原始与可靠状态输入指纹不变。
下一轮仅预设在同一128条/相同外折中比较新增质量比前后；当前训练入口仍仅接受旧两核心列，需另建受证据门禁约束的接入版本，不能直接改旧配置绕过。

## 独立接入细骨料比并进行同折比较（用户继续后已执行）

```bash
/workspace/.venv-underwater-ml/bin/python scripts/run_evidence_training.py --output outputs/new-fine-ratio-training
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

新增独立入口`evidence_train.py`，原两核心输入入口及历史配置未修改。它验证严格范围128条身份、原对照网格及外折、证据清单/快照指纹，重新读取原PDF并重建质量比，确认与候选输入完全相同后才拟合。新增三列变换器置于Pipeline中；缩放和MLP目标缩放仍在每个内训练折拟合。
实际新增12项实验、60个外层验证折、1536条折外预测，结果在`outputs/fine-ratio-train-only-v1`。与原12项实验逐条配对的Excel及报告在`outputs/fine-ratio-comparison-v1`，汇总脚本`summarize_fine_ratio_training.py`使用这两个实际目录。
SVR在保留计算水灰比的方案中MAE从11.511降至7.503MPa，但R²仍为−0.0230，来源组等权MAE23.729MPa，5组中3组改善、2组变差。所有新增方案R²均为负；没有将MAE下降直接解释为模型可靠或所有来源获益。
`paired_error_changes`以相同ID和暂定目标配对，分开计算样本加权及来源组等权变化；均值配置逐条预测未变，外折与原预案一致。全量负预测和大误差保留，145条测试没有预测或评价。
新增6项测试，完整86项通过；原数据与可靠状态指纹不变，新增可信资格0，未选择或部署最终模型。

## 胶材、纤维与原文材料家族诊断

```bash
# 首次编录；已有配置时跳过此行
python scripts/prepare_composition_review.py
/workspace/.venv-underwater-ml/bin/python scripts/run_composition_review.py --output outputs/new-composition-review
PYTHONPATH=src /workspace/.venv-underwater-ml/bin/python -m unittest discover -s tests -v
```

最终输出为`outputs/composition-review-v2/report.md`及`胶材纤维与材料体系核查.xlsx`。严格训练128条/5来源组，分别保存明确值、原表横线及未列项目；未用未知0补齐。粉煤灰比16条/3组、硅灰7条/2组、矿渣4条/1组；钢/PVA/PP纤维比各4/6/1条，不把不同纤维效果或体积分数视为等价。
原文声明家族共5类，每类仅1独立来源组，全部128条在留来源验证时家族类别都未在训练折出现；这些自述家族/用途标签并非互斥通用分类，本轮保持诊断用途。
定义内胶材组分均明确、可完整求和的只有11条；水胶比分项候选10条中Al-Selmi六条总量矛盾仍未解决。Zhang用归一化组分求和不核实原库绝对用量；Yousri和Brzozowski共14条原Water单位L/m³，未自动转换kg/m³或回写历史核心输入。
`complete_component_sum`遇未知即不求完整和；`derive_composition`保存分项和/原文总量冲突及类型独立比例；`family_confounding`检查原文家族与来源混杂。新增5项测试，完整91项通过；未重训、未用测试行、未新增可信资格。
