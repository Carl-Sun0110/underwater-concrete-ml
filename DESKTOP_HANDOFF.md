# 桌面Codex任务交接（2026-10-25）

## 目标与起点

继续水下浇筑混凝土28天抗压强度预测研究。实际Baseline、诊断和CSV预测已经跑通，不要从总体方案重新开始，也不要优先搭复杂Agent。文献核查与探索建模并行。

仓库Carl-Sun0110/underwater-concrete-ml。用户已授权整理迁移GitHub。代码、配置、最新汇总和预测模型在main；原始材料及完整历史输出的4个ZIP在独立`desktop-transfer-20261025`分支，方便绕过聊天下载故障。

1. 先读README、本文件与`results/latest/阶段性探索建模报告.md`。
2. 安装`requirements-desktop.txt`，先用`examples/recipes_demo.csv`跑预测。
3. 在main仓库运行`python scripts/download_migration.py --extract`，按SHA256取包且仅恢复历史输出和材料；之后运行`python scripts/prepare_desktop.py`。本地配置目录已存在时用`--output-dir configs/local_restored`创建新的配置，不能覆盖已编辑配置。不能按同名猜版本。
4. 要重跑时使用`configs/local/`配置，输出`outputs/desktop_baseline_v1`，然后V1.1。保留原始历史配置和manifest。

## 数据事实

- 原版Total1082条、水下710条、水下Converted正值582条。
- 主集212条；扩展467条（含主集和255条范围待核查）；30条已知非直接成型记录隔离。不是527条全部可信，也不是212条已正式全字段核实。
- 请求过的《水下混凝土_人工核查版.xlsx》《水下混凝土_筛选结果_20261012.xlsx》始终未找到。实际另有原版Selected、批注核对版及筛选方案，不能冒充所缺版本。
- Selected为编码/零填版本，不直接当原始建模表。
- 原始材料16个：4份Excel、11份PDF、1份用户粘贴文本；逐文件哈希和原云路径见`migration/materials_manifest.json`。ZIP为短可读文件名，内容字节未改。
- AWA为抗冲散剂，RS为河砂，SS为硅砂；不要将RS解释为再生骨料、AWA解释为消泡剂。
- 用户已许可按0.95、0.8、0.76暂定系数探索，不要重复询问许可；系数原文依据仍待核，算术一致不授信。

## 模型和验证

Baseline V1固定RF/XGBoost/CatBoost参数、种子42，隔离近重复配比并按论文/研究组及跨论文近重复连通组验证。主集随机R²较高、论文留出R²均负，不能宣布高精度跨文献泛化。

V1.1仅用固定论文留出后的训练记录：主集117条/4来源组，扩展320条/23来源组。95/147条留出后续未重评。主集Nasr104条占117条大多数，分组CV很不稳定。扩展候选为Water/Cem＋中位数XGBoost：训练CV MAE9.118、RMSE12.227 MPa、R²0.084、来源宏平均MAE10.668。`models/exploratory_v1_1.joblib`是已训练模型原样复制。

来源重采样诊断固定训练折外误差：23组中12组优于均值、11组更差；误差差值范围跨0，非独立测试置信区间或p值。选模与CV复用、多轮探索均有适应性偏差。不要把不同队列、目标和划分的最低分混成最终成绩。

## 优先核查

1. **Zhongzhe Zhang 2023**：*Effect of antiwashout admixture on compressive strength of alkali-activated underwater non-dispersible concrete*。原表DOI`10.1680/jmacr.22.00239`，本轮未联网核验。747/751/755/759行Cem0、GGBS550、AlkaliAct189，可能为无水泥体系，需PDF确认。751行Water64、相关其他行128，核对激发剂溶液/有效水，不能按邻行改值。
2. **Wang 2024**：790行涉及砂浆灌入预置骨料，确认强度标签对象。原表备注/CastMethod仅为线索。
3. **Horszczaruk 2017**：原表备注称Underwater为推定；2023同作者论文的压力养护/成型条件也待核。
4. **Zhang 2025、Al-Selmi 2026**：已复查相对水泥质量基准和分项1503/表注1502冲突，均保留。不以假设密度反推绝对用量，高误差不直接判为错标。

已整理4篇/26条核查包。指定4条Cem0照录假设使核心XGBoost训练CV MAE9.118→9.018，不能证实真零；原始数据、标签、正式状态、CSV零值规则及发布模型均未修改。当前代理403，新PDF未取得。

## 执行约束

- 原始材料只读，新实验输出新目录，保留失败结果。
- 不补造/反推目标，不用其他实测性能或目标派生字段作输入；Density须确认是设计可获参数。
- 插补、缩放、筛选、调参限训练折，保持论文/配比连通组隔离，不为高分删困难样本。
- 待核查可单列探索实验，不以机械计算授信。记录版本、核查状态与处理方法。
- 教师汇报文字只是本地文件，未自动发送任何消息。

## 环境兼容

已验证Python3.12/Linux；完整历史复现推荐WSL/Linux。原生Windows完整训练未实测。旧脚本/配置/清单含`/workspace/...`，一些历史集成测试依赖云路径及冻结哈希，不能直接声称桌面全套通过。

`prepare_desktop.py`只生成新的本地V1/V1.1配置，不重签历史清单。后续旧审计/神经网络/多阶段管线需另建本地配置。原始归档保留云端结果原貌；不要把路径问题当作模型结论。

## 关键文件

- `src/concrete_audit/baseline_v1.py`：只读导入、筛选、重复识别、三模型固定协议。
- `baseline_refinement.py`：训练内插补/特征对照。
- `predict.py`、`scripts/predict_strength.py`：白名单CSV预测及范围标记。
- `evidence_batch.py`：原表核查包与Cem0假设。
- `migration/scope_registry_snapshot.json`：派生范围索引，状态未改；完整引用在历史归档，不当作完整原文证据。
- `results/latest/`：精简报告；`outputs/`：离线恢复的完整历史，未入Git。
- `docs/CLOUD_WORKLOG_README.md`与CHANGELOG：历史记录，早期阶段说明可能被后续用户要求取代。

桌面Codex起始指令：

> 请先读DESKTOP_HANDOFF.md。保留Baseline与待核查状态，核对本地材料哈希并生成新配置，实际运行CSV预测。材料已恢复时按本地新配置验证V1流程；Zhongzhe Zhang 2023原文已提供时优先逐表核对4条Cem0记录。不重复总体设计，不擅改原始标签。
