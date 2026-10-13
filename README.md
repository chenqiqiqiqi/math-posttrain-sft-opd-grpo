# GPU 时间预算约束下的数学推理后训练对比：SFT / OPD / GRPO

复现并扩展 [Rethinking On-Policy Distillation of LLMs](https://arxiv.org/abs/2604.13016)（代码：[Thinking-Space/Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD) @ `ac26e38`）。

研究问题：在同一 GPU 时间预算下，离线蒸馏（SFT）、在线策略蒸馏（On-Policy Distillation, OPD）与 GRPO 对 1.5B 推理模型的单次成功率（pass@1）和多次采样覆盖率（pass@k）有何影响；并复现论文中"从底座直接 OPD 失败、SFT 冷启动后可救回"的现象。

## 主要结果

主口径：AIME24/25/26 + AMC23 共 130 题，剔除与教师已知训练集（DAPO-Math-17K）重合的 29 题后为 **101 题**；每题采样 32 次；逐题配对 bootstrap 95% CI（1 万次）+ 符号检验。

| 模型 | pass@1 | pass@32 | 32 次全错题数 | 截断率 | 平均长度 |
|---|---|---|---|---|---|
| D0（学生，DeepSeek-R1-Distill-Qwen-1.5B） | 30.6 | 62.4 | 38 | 8.9% | 15,079 |
| D-GRPO（预算内 92 步） | 31.0 | 68.3 | 32 | 3.0% | 12,324 |
| D-SFT | **44.8** | 72.3 | 28 | 3.8% | 10,099 |
| D-OPD（150 步） | **43.7** | 72.3 | 28 | 0.6% | 10,301 |
| D-T（教师，JustRL-DeepSeek-1.5B） | 47.7 | 73.3 | 27 | 1.7% | 9,576 |

| 配对比较（A − B，百分点） | pass@1 | pass@32 |
|---|---|---|
| D-SFT − D0 | **+14.2 [+10.3, +18.3]** | **+9.9 [+4.0, +16.8]** |
| D-OPD − D0 | **+13.1 [+9.4, +16.8]** | **+9.9 [+4.0, +16.8]** |
| D-GRPO − D0 | +0.4 [−1.2, +2.1] | +5.9 [0.0, +12.9]（9 胜 3 负，p = 0.146） |
| D-OPD − D-SFT | −1.2 [−3.4, +1.1] | 0.0 [−4.0, +4.0] |

- 本设置下 **SFT ≈ OPD ≫ GRPO（pass@1）**。SFT 与 OPD 的差异不显著。
- 同预算下 GRPO 未见显著提升。记录到的情况：GRPO 每步约 760 秒（OPD 约 463 秒），预算内完成 92 步；开发集 +3.4，竞赛题 +0.4。更长训练是否改变结论尚未验证。
- pass@32：SFT/OPD 各 +9.9；GRPO 点估计 +5.9，未见显著提升；三种方法之间差异不显著。原模型 32 次全错的 38 题中，SFT / OPD / GRPO 分别解出 11 / 11 / 9 题——这是固定 32 次采样下的可观测覆盖率变化，不等于证明原模型"不会"这些题。
- 截断：新解出的题上原模型截断比例偏高（21.6%，全部全错题为 12.4%），但完整写完却答错的回答仍占多数（约 78%）。截断因素的具体贡献尚未做敏感性分析。

### 复现：直接 OPD 失败 → SFT 冷启动救回（Q 族）

学生 Qwen3-1.7B-Base，教师 Qwen3-4B（非思考）。冷启动模型为**论文作者发布的 SFT 模型**（非本项目训练）。每题 16 次采样，最长 8,192 token。

| 模型 | avg@16 | pass@16 | 截断率 |
|---|---|---|---|
| Q-Base | 3.9 | 21.8 | 41.4% |
| Q-OPD（底座直接 OPD，40 步） | 5.1 | 23.8 | 63.9% |
| Q-SFT（作者的冷启动模型） | 12.9 | 31.7 | 20.9% |
| Q-SFT-OPD（冷启动后 OPD，40 步） | **15.2** | 31.7 | 10.3% |

- 事先写定的两条复现判据均满足：(a) 前 10 步学生–教师 top-16 token 重合率 0.716 > 0.582；(b) Q-SFT-OPD − Q-OPD = +10.0 [+6.4, +13.8]。
- 这 +10.0 主要来自冷启动起点；冷启动后追加 40 步 OPD 相对起点本身为 +2.2 [+0.8, +3.8]。
- 直接 OPD 的训练过程：重合率停滞在约 0.6 并回落，熵 2.3 → 3.4，回答长度 1,340 → 4,620 token，截断 5% → 46%。

## 实验协议要点

- **预算**：B = D-OPD 跑满 150 步的实测墙钟 × 2 卡 = 69,528 秒 × 2 = **38.63 GPU·h**（2 × RTX 4090D 48GB）。计费从第一次生成到最后一次参数更新，含保存检查点，不含故障损失。
- **按预算停止**：给 verl 的 `RayPPOTrainer.fit` 打补丁（[patches/verl_budget_stop.patch](patches/verl_budget_stop.patch)），每步记账，累计 ≥ B 时保存并结束（最多超一步）。D-GRPO 实际为 B 的 1.006 倍。
- **SFT 的预算**：SFT 无法中途截停（cosine 调度依赖总步数），按试跑吞吐量预先确定数据量、训练 1 轮；含教师生成在内实际为 B 的 **1.040 倍**，如实报告。
- **事先冻结**：训练前冻结协议（v1 + 补充 1–9，每条注明时间及是否在看到相关结果之前）；测试集在全部模型冻结后只评一次；开发集（DeepMath 200 题 × 8）只画学习曲线，不参与选检查点。
- **评测流程校验**：教师在本流程下 AIME24 52.6 / AIME25 37.8，JustRL 论文报告 52.60 / 38.75；学生 MATH-500 84.3%（模型卡 84.9%）。

完整协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，完整结果与解读见 [reports/results/SUMMARY.md](reports/results/SUMMARY.md)。

## 局限

- 单种子（seed 42）；1.5B / 1.7B 学生、单一预算，结论只适用于本设置。
- D 族教师同为 1.5B；SFT 与 RL 训练器效率不同，"同 GPU 时间"限定为本实现下。
- Q 族在 8K 上限下截断率较高（截断按答错计），可能低估需要长推理的情况；Q 族与 D 族评测长度不同，不做跨族比较。
- 学生模型训练数据未公开，无法对其做去污染检查。

## 仓库结构

```
EXPERIMENT_PROTOCOL.md      实验协议（v1 冻结 + 补充 1–9）
patches/                    对上游 verl 的修改（墙钟预算停止）
scripts/                    数据构建、训练、评测、统计脚本
  build_data.py             构建训练/开发/测试集
  train_verl.sh, run_arm.sh OPD / GRPO 训练与单组流水线
  sft_gen.py, train_sft.sh, run_sft_arm.sh   SFT：教师生成 + 训练
  eval_generate.py, eval_grade.py, eval_jobs.py   vLLM 生成、判分、双卡评测排队
  stats.py, report.py       pass@k 无偏估计、配对 bootstrap、符号检验、结果表
manifests/                  模型 revision、数据来源与哈希、依赖版本、教师重合题名单
reports/
  results/                  结果表（test_results.md/json）与解读（SUMMARY.md）
  decontam/                 去污染报告与相似度明细
  pilot/                    试跑记录
results/
  eval/{dev,test}/<模型>/graded.csv   逐题正确次数、截断次数、平均长度
  train/<组>/                训练日志、tensorboard、预算台账
```

模型权重与生成原文较大，未放入仓库。
