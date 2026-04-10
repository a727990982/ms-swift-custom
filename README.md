# ms-swift GRPO TensorBoard By Key

这个仓库提供了一个很轻量的 GRPO 外部 plugin，用来把 TensorBoard 指标按样本字段拆开看。

设计目标只有两件事：

- 不改上游 `ms-swift` 源码
- 让 `overall` 和按 key 拆分的 grouped run 尽量保持同一统计口径

## 用法

在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/ms-swift-3.12.0:${PYTHONPATH}"

python3 scripts/run_grpo_with_tb_key.py \
  --tb_group_key task \
  --external_plugins custom_plugins/grpo_tensorboard_by_key.py \
  --logging_dir output/exp1/overall \
  --rlhf_type grpo \
  ...其余 ms-swift 参数保持不变...
```

说明：

- `run_grpo_with_tb_key.py` 只负责把 `--tb_group_key` 转成环境变量
- plugin 需要通过 `--external_plugins` 显式传入
- `swift` 需要通过 `PYTHONPATH` 显式暴露出来
- 如果你不想在 TensorBoard 里看到整体 run 名字是 `.`, 推荐把 `--logging_dir` 设成 `.../overall`

## TensorBoard 目录

推荐目录结构：

```text
output/exp1/
├── overall/
│   └── events.out.tfevents...
└── grouped/
    ├── task=fast/
    │   └── events.out.tfevents...
    └── task=slow/
        └── events.out.tfevents...
```

查看方式：

```bash
tensorboard --logdir output/exp1
```

## 当前会拆分的指标

plugin 目前会按 key 写出这些 grouped 指标：

- `reward`
- `reward_std`
- `frac_reward_zero_std`
- `rewards/<name>/mean`
- `rewards/<name>/std`
- `completions/mean_length`
- `completions/min_length`
- `completions/max_length`
- `completions/clipped_ratio`
- `entropy/mean`
- `entropy/min`
- `entropy/max`
- `num_turns`
- `samples`

其中：

- `samples` 是 grouped 额外提供的辅助指标，overall 里没有对应项
- `kl`、`entropy/threshold`、`rollout_correction/*`、clipping 相关指标目前不会按 key 拆分
- `lr`、`grad_norm` 这类全局优化器指标也不会拆分

## 对齐说明

当前 grouped 指标的实现已经尽量跟随 `ms-swift` 原生 overall 指标：

- 使用同一个 `log()` 窗口
- `completion` 和 `reward` 类指标按 rollout 级标量记录后再聚合
- `entropy` 类指标按 loss 计算时的 batch 级标量记录后再聚合

因此日常最常看的 grouped 曲线可以直接和 overall 对比。

仍需注意：

- 如果你已经生成过旧的 TensorBoard event 文件，旧曲线不会被自动修正
- 在极少数 `dynamic_num_samples` 触发 chunked loss 的场景下，`entropy` 极值的展示仍可能和 overall 存在轻微差异

## 仓库结构

```text
.
├── custom_plugins/
│   └── grpo_tensorboard_by_key.py
├── scripts/
│   └── run_grpo_with_tb_key.py
└── ms-swift-3.12.0/
```
