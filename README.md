# 水下混凝土28天抗压强度预测

项目已完成只读导入、范围标记、缺失处理、RF/XGBoost/CatBoost基线、配比簇随机/论文分组评价及CSV预测。文献核查和探索建模并行，当前结果不是最终可靠科研结论。

**桌面Codex接手先读 [DESKTOP_HANDOFF.md](DESKTOP_HANDOFF.md)。** 历史长记录保留在[云端工作记录](docs/CLOUD_WORKLOG_README.md)，其中早期门槛不代表当前建模前置条件。

## 当前状态

- Total1082条，水下710条，水下Converted正值582条；主集212条，扩展探索集467条。主集辅助证据支持成型范围，整条数据及换算仍待核。
- V1.1扩展训练320条/23来源组，147条论文留出不在后续实验重评。Water/Cem＋中位数XGBoost训练CV：MAE9.118、RMSE12.227 MPa、R²0.084。
- 23来源组中12组优于均值参照、11组更差。随机分数不能替代跨文献泛化能力。
- 0.95/0.8/0.76为用户已许可暂定系数；不插补目标、不覆盖原始材料、不按误差删困难样本。

最新报告见[results/latest](results/latest)，模型见[models](models)。主分支不保存大型原始PDF/Excel；迁移ZIP暂存专用`desktop-transfer-20261025`分支，不合并主分支。

## 桌面运行

云端已验证Python3.12/Linux；完整研究复现推荐同环境或WSL，原生Windows完整训练未实测。依赖按平台选择XGBoost包。

```bash
python -m venv .venv
# Linux/WSL: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-desktop.txt
```

Linux/WSL：

```bash
PYTHONPATH=src python scripts/predict_strength.py --model models/exploratory_v1_1.joblib --input examples/recipes_demo.csv --output outputs/desktop_demo_predictions.csv
```

PowerShell先执行`$env:PYTHONPATH="src"`，再执行上述`python scripts/predict_strength.py ...`部分。输出文件须不存在。示例是训练样本回放，不是新准确率证据。

CSV列为`case_id,Water,Cem`，单位沿用原库名义基准，不自动进行水体积/质量或密度转换。缺失和超训练范围会标记；范围内也不证明可靠，未提供置信区间。

## 恢复材料与研究流程

聊天下载不可用时，在仓库根目录直接通过Git取包并恢复，无需浏览器分别下载：

```bash
git pull --ff-only origin main
python scripts/download_migration.py --extract
```

脚本校验4包SHA256，只恢复`outputs/`和`data/raw/attachments/`，保留当前代码及本地不同内容文件，不改历史哈希。下载结果见`migration_archives/download_report.json`。也可从迁移分支网页下载4个ZIP。已有本地修改不应被手动解压覆盖。

```bash
python scripts/prepare_desktop.py
```

工具按原始字节SHA256匹配材料，生成`configs/local/desktop_setup_report.json`和新本地配置，不改旧配置或历史哈希。材料在别处时加`--materials-root <路径>`；已有本地配置时用`--output-dir`另选目录。预测不依赖原始材料；重跑Baseline至少需要原版Total。

```bash
PYTHONPATH=src python scripts/run_baseline_v1.py --config configs/local/baseline_v1.json --output outputs/desktop_baseline_v1
PYTHONPATH=src python scripts/run_baseline_v1_1.py --config configs/local/baseline_v1_1.json --output outputs/desktop_baseline_v1_1
```

先完成本地V1，再运行V1.1。历史管线仍含云端绝对路径；后续重跑需继续生成指向本地前序结果的新配置，不能批量改写旧manifest来伪称复现。

## 验证

未恢复材料时先运行独立测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_desktop_setup.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_stability.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_audit.py -v
```

全套旧集成测试依赖冻结云路径、数据和哈希，桌面直接运行可能存在路径依赖。不要声称复制归档即等于所有测试已通过；应逐步拆分本地集成测试。

## 下一步

优先取得Zhongzhe Zhang等2023年 *Effect of antiwashout admixture on compressive strength of alkali-activated underwater non-dispersible concrete*，原表DOI`10.1680/jmacr.22.00239`，本轮未联网核验。核对747/751/755/759行Cem0、激发剂和水量定义。当前代理403，PDF仍未取得；标题、原表线索和模型分数都不能当作原文已核实。
