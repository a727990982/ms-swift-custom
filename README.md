# ms-swift Local Tools

这个仓库目前放了两个 repo-local 的 ms-swift 扩展：

- 一个很轻量的 GRPO 外部 plugin，用来把 TensorBoard 指标按样本字段拆开看
- 一个非侵入式的 vLLM infer 猴子补丁入口，用来把数据集推理改成边编码边投喂 vLLM

设计目标只有两件事：

- 不改上游 `ms-swift` 源码
- 让 `overall` 和按 key 拆分的 grouped run 尽量保持同一统计口径

## vLLM Infer 猴子补丁入口

如果你想保留上游 `ms-swift` 源码不变，但又想优化 `swift infer --infer_backend vllm` 的数据集推理链路，可以使用：

```bash
python3 scripts/run_vllm_infer_patched.py \
  --infer_backend vllm \
  --model Qwen/Qwen2-7B-Instruct \
  --val_dataset AI-ModelScope/alpaca-gpt4-data-zh#1000 \
  --patched-vllm-encode-num-workers 8 \
  --patched-vllm-infer-prefetch-size 256
```

说明：

- 这个脚本不会改 `ms-swift-3.12.0/swift/**` 下的源码，而是在当前 Python 进程里先打猴子补丁，再调用原生 `infer_main(args)`
- 它会自动把仓库里的 `ms-swift-3.12.0` 和 repo 根目录加入 `sys.path`，不依赖你手工设置 `PYTHONPATH`
- 只在 `infer_backend=vllm` 时启用补丁；其他后端仍走原始 `infer_main`
- 只在数据集推理场景下默认把 `vllm_use_async_engine` 提升为 `True`
- 如果你显式传了 `--vllm_use_async_engine false/true`，脚本不会覆盖你的设置
- `--patched-vllm-encode-num-workers` 和 `--patched-vllm-infer-prefetch-size` 是 wrapper 自己的参数，不会传给上游 ms-swift dataclass

脚本兼容大部分原 `swift infer` 参数，例如：

- `--dataset` / `--val_dataset`
- `--cached_val_dataset`
- `--load_data_args`
- `--config xxx.yaml`

其中 `write_batch_size` 仍然只负责结果分片写盘，不再要求整片样本先编码完才进入 vLLM。

如果你想直接验证这条链路，可以运行现成脚本：

```bash
bash scripts/test_vllm_infer_qwen3vl.sh
```

这个脚本会：

- 默认使用 `Qwen/Qwen3-VL-4B-Instruct`
- 自动生成一个本地的多模态 jsonl 验证集
- 调用 `scripts/run_vllm_infer_patched.py` 走 vLLM 数据集 infer 路径
- 允许你通过环境变量覆盖 `MODEL`、`SAMPLE_COUNT`、`CUDA_VISIBLE_DEVICES`、`VLLM_TP` 等参数

## GRPO TensorBoard By Key

这个功能对应下面这组文件：

- `scripts/run_grpo_with_tb_key.py`
- `custom_plugins/grpo_tensorboard_by_key.py`

### 用法

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

### TensorBoard 目录

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

### 当前会拆分的指标

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

### 对齐说明

当前 grouped 指标的实现已经尽量跟随 `ms-swift` 原生 overall 指标：

- 使用同一个 `log()` 窗口
- `completion` 和 `reward` 类指标按 rollout 级标量记录后再聚合
- `entropy` 类指标按 loss 计算时的 batch 级标量记录后再聚合
- 多卡场景下优先按“完整样本记录”做 gather，避免 group 字段和数值张量错位
- `chunked loss` 场景下会同步切分 `tb_group_values`，避免 grouped entropy 在分 chunk 时错位

因此日常最常看的 grouped 曲线可以直接和 overall 对比。

仍需注意：

- 如果你已经生成过旧的 TensorBoard event 文件，旧曲线不会被自动修正
- 如果运行时出现 `Skipped grouped ...` 警告，表示当前批次的分组元数据和上游日志长度不一致；这类批次会被跳过，不会静默写入错误指标

## 仓库结构

```text
.
├── custom_plugins/
│   ├── grpo_tensorboard_by_key.py
│   └── vllm_infer_pipeline_patch.py
├── scripts/
│   ├── run_grpo_with_tb_key.py
│   └── run_vllm_infer_patched.py
└── ms-swift-3.12.0/
```
