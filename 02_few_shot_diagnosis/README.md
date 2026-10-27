# 小样本故障诊断（凌晨）

本方向完成公共质量处理、过程特征、RF/CNN、任务式ProtoNet元学习、预训练后微调的迁移学习及独立算子封装。当前证据来自 **RCSD公开压缩机运行背景＋规则构造异常**，不是现场真实故障标签。真正跨数据集迁移、至少8类真实故障覆盖和现场验收精度仍待验证。

| 材料 | 位置 |
|---|---|
| 群发版实验报告 | [docs/凌晨/实验报告.md](../docs/凌晨/实验报告.md) |
| 技术报告及失败尝试 | [docs/凌晨/技术报告.md](../docs/凌晨/技术报告.md) |
| 算子参数、调用与部署 | [docs/凌晨/算子说明.md](../docs/凌晨/算子说明.md) |
| 数据、划分和复现 | [docs/凌晨/数据与复现.md](../docs/凌晨/数据与复现.md) |
| 本次上传测试记录 | [docs/凌晨/测试报告.md](../docs/凌晨/测试报告.md) |
| 10个独立算子与仓库登记 | [dataflow_operators](../dataflow_operators/README.md) |
| 39份模型、契约、校验清单 | [resources/rcsd_v2](resources/rcsd_v2/README.md) |
| 主实验指标、预测和日志 | [results/rcsd_v2](results/rcsd_v2) |
| 未采用增强方案记录 | [results/unsuccessful_trials](results/unsuccessful_trials) |

## 安装与快速运行

验证环境：Python 3.12，精确依赖见 [requirements.txt](requirements.txt)。推荐独立虚拟环境。

```text
python -m pip install -r 02_few_shot_diagnosis/requirements.txt
python 02_few_shot_diagnosis/scripts/run_rcsd_demo.py
python tools/verify_rcsd_handover.py
```

快速演示使用人工数列检查接口，不是公开实测数据，不据此计算准确率。已有模型无需重新训练即可运行。输出默认保存在Git忽略的 `outputs/`。

仅复用清洗/补缺/对齐：`python -m pip install -r requirements-public-processing.txt`，不需要安装PyTorch。

## 数据与主要结果

原始公开工作簿、派生传感器窗口及任何现场数据不随Git仓库分发。按[数据与复现](../docs/凌晨/数据与复现.md)自行取得公开工作簿并验证SHA256后，可恢复520窗并复核模型。

25测点、15分钟间隔、20点窗（首末点4.75小时）。整背景划分：1—6月31背景供源训练，7—8月11背景供目标支持池，9—12月22背景供查询。每背景构造4状态，源标签124窗；每类K=1/5/10时目标标注4/20/40窗，不是同等数量的独立故障事件。

每类5个目标标注、三次支持抽样的主场景平均准确率：RF 85.61%，CNN集成88.64%，元学习适配集成90.53%，迁移集成88.26%。迁移仍有背景误判，未证实优于对应从头CNN。固定支持抽样11的元学习集成案例为89.77%，不能替代90.53%的重复平均。

## 开发与检查

共享实现位于 `common_algorithms/`；训练与划分在 `models/`、`pipelines/`；复现入口在 `scripts/`。推理不进行在线训练。数据、支持标签和最终查询严格分开，物理输入契约必须匹配。

```text
python tools/build_rcsd_operators.py
python -m pytest tests/public_processing 02_few_shot_diagnosis/tests 03_cross_condition/tests -q
```

实际平台注册、卡片生成和平台依赖/模型挂载由维护方按新规范完成；仓库登记和本地执行检查不代替平台联调。
