# ms-swift GRPO TensorBoard By Key

一个基于 `ms-swift` 的小型扩展工程，用来在 GRPO 训练时，按照数据样本中的某个字段分类监控 TensorBoard 指标。

当前仓库采用“非侵入式”方式集成：

- 不修改上游 `ms-swift` 原始源码
- 通过 `--external_plugins` 在运行时注册外部 plugin 并 patch `GRPOTrainer`
- 通过独立启动脚本传入分类 key

## 目录结构

```text
.
├── custom_plugins/
│   └── grpo_tensorboard_by_key.py
├── scripts/
│   └── run_grpo_with_tb_key.py
└── ms-swift-3.12.0/
```

## 功能

- 支持通过参数指定样本分组字段，例如 `task`、`domain`、`source`
- 在原有 TensorBoard 日志目录内，额外写入按类别拆分的 GRPO 指标
- 优先读取样本顶层字段；若不存在，也支持从 `data_dict` 中读取
- 保持上游训练流程不变，便于后续升级或替换 `ms-swift`

## 使用方式

在本仓库根目录运行：

```bash
export PYTHONPATH="$PWD/ms-swift-3.12.0:${PYTHONPATH}"

python3 scripts/run_grpo_with_tb_key.py \
  --tb_group_key your_key \
  --external_plugins custom_plugins/grpo_tensorboard_by_key.py \
  --rlhf_type grpo \
  ...其余 ms-swift GRPO 参数保持不变...
```

例如：

```bash
export PYTHONPATH="$PWD/ms-swift-3.12.0:${PYTHONPATH}"

python3 scripts/run_grpo_with_tb_key.py \
  --tb_group_key task \
  --external_plugins custom_plugins/grpo_tensorboard_by_key.py \
  --logging_dir output/exp1/overall \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset your_dataset \
  --rlhf_type grpo \
  --reward_funcs format
```

说明：

- 这个脚本不会自动注入 plugin
- 这个脚本不会自动注入 `ms-swift` 源码目录
- 需要先通过 `export PYTHONPATH="$PWD/ms-swift-3.12.0:${PYTHONPATH}"` 让 Python 能找到 `swift`
- 需要你显式通过 `--external_plugins custom_plugins/grpo_tensorboard_by_key.py` 传给 `ms-swift`
- 如果你希望 TensorBoard 里的整体 run 名字不是 `.`, 推荐把 `--logging_dir` 设成形如 `.../overall`

## TensorBoard 查看方式

写入结构示例：

```text
output/exp1/
├── overall/
│   └── events.out.tfevents...          # 整体指标，保持 ms-swift 原始视图
└── grouped/
    ├── task=math/
    │   └── events.out.tfevents...
    └── task=code/
        └── events.out.tfevents...
```

每个单独 key run 里的标量名称示例：

```text
train/reward
train/reward_std
train/rewards/format/mean
train/completions/mean_length
eval/...
```

这样在 TensorBoard 里可以分别看：

- 整体：`overall`
- 每个 key：`grouped/task=math`、`grouped/task=code` 这类单独 run

推荐启动方式：

```bash
tensorboard --logdir output/exp1
```

## 说明

- 这套实现更适合“与样本强相关”的 GRPO 指标，例如 reward、reward_std、completion length、entropy 等
- 像 `lr`、`grad_norm` 这类全局优化器指标，不会按类别拆分
- `ms-swift-3.12.0/` 目录当前作为上游源码快照保存在仓库中

## 后续建议

- 初始化首个 commit
- 配置 GitHub 远端并推送
- 按你的实际训练命令补一个可直接运行的示例脚本
