import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate.utils import gather_object

from swift.utils import get_logger

logger = get_logger()

ENV_KEY = 'SWIFT_GRPO_TB_GROUP_KEY'
MISSING_VALUE = '__missing__'
PLUGIN_TAG = '_tb_group_metrics_plugin_enabled'


def _stringify_group_value(value: Any) -> str:
    if value is None:
        return MISSING_VALUE
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    return text or MISSING_VALUE


def _extract_group_value(row: Dict[str, Any], key: str) -> str:
    if key in row:
        return _stringify_group_value(row.get(key))
    data_dict = row.get('data_dict')
    if isinstance(data_dict, dict) and key in data_dict:
        return _stringify_group_value(data_dict.get(key))
    return MISSING_VALUE


def _sanitize_tag(value: str) -> str:
    value = str(value).strip().replace('/', '_').replace('\\', '_').replace(' ', '_')
    value = value[:120]
    return value or MISSING_VALUE


def _last_indices(values: List[Optional[str]]) -> List[int]:
    last_pos = {}
    for idx, value in enumerate(values):
        last_pos[value] = idx
    return sorted(last_pos.values())


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values) or 0.0
    var = sum((x - mean)**2 for x in values) / (len(values) - 1)
    return float(var**0.5)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # nan
        return None
    return value


def _new_group_window() -> Dict[str, Any]:
    return {
        'group_value': [],
        'prompt_id': [],
        'request_id': [],
        'completion_length': [],
        'truncated': [],
        'num_turns': [],
        'entropy': [],
        'rewards': defaultdict(list),
    }


def _clear_group_window(window: Dict[str, Any]) -> None:
    window['group_value'].clear()
    window['prompt_id'].clear()
    window['request_id'].clear()
    window['completion_length'].clear()
    window['truncated'].clear()
    window['num_turns'].clear()
    window['entropy'].clear()
    window['rewards'] = defaultdict(list)


def _get_group_window(self, mode: Optional[str] = None) -> Dict[str, Any]:
    if mode is None:
        mode = 'train' if self.model.training else 'eval'
    return self._tb_group_windows[mode]


def _window_sample_count(self, window: Dict[str, Any]) -> int:
    lengths = [
        len(window['group_value']),
        len(window['prompt_id']),
        len(window['request_id']),
        len(window['completion_length']),
        len(window['truncated']),
        len(window['num_turns']),
    ]
    for reward_name in self.reward_func_names:
        lengths.append(len(window['rewards'][reward_name]))
    return min(lengths) if lengths else 0


def _group_run_name(group_key: str, group_value: str) -> str:
    safe_key = _sanitize_tag(group_key)
    safe_value = _sanitize_tag(group_value)
    return f'{safe_key}={safe_value}'


def _resolve_group_run_root(logging_dir: str) -> Path:
    logging_dir = Path(logging_dir)
    if logging_dir.name == 'overall':
        return logging_dir.parent / 'grouped'
    return logging_dir / 'grouped'


def _get_group_writer(self, group_value: str):
    writer = self._tb_group_writers.get(group_value)
    if writer is not None:
        return writer

    run_name = _group_run_name(self._tb_group_key, group_value)
    log_dir = self._tb_group_run_root / run_name
    writer = self._tb_group_summary_writer_cls(log_dir=str(log_dir))
    self._tb_group_writers[group_value] = writer
    return writer


def _build_rows(self, window: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    seen_nums = _window_sample_count(self, window)
    if seen_nums == 0:
        return rows

    group_values = window['group_value'][:seen_nums]
    prompt_ids = window['prompt_id'][:seen_nums]
    request_ids = window['request_id'][:seen_nums]
    completion_lengths = window['completion_length'][:seen_nums]
    truncated = window['truncated'][:seen_nums]
    num_turns = window['num_turns'][:seen_nums]
    entropy_logs = window['entropy'][:seen_nums] if len(window['entropy']) == seen_nums else []
    reward_values = {
        name: window['rewards'][name][:seen_nums]
        for name in self.reward_func_names
    }

    weights = self.reward_weights.detach().float().cpu().tolist()
    for idx in range(seen_nums):
        reward_items = {}
        total_reward = 0.0
        valid_reward = False
        for reward_name, weight in zip(self.reward_func_names, weights):
            reward_value = _safe_float(reward_values[reward_name][idx])
            reward_items[reward_name] = reward_value
            if reward_value is not None:
                total_reward += reward_value * float(weight)
                valid_reward = True

        entropy = entropy_logs[idx] if idx < len(entropy_logs) else None
        rows.append({
            'group_value': group_values[idx],
            'prompt_id': prompt_ids[idx],
            'request_id': request_ids[idx],
            'completion_length': _safe_float(completion_lengths[idx]),
            'truncated': bool(truncated[idx]),
            'num_turns': _safe_float(num_turns[idx]),
            'entropy': _safe_float(entropy),
            'rewards': reward_items,
            'total_reward': total_reward if valid_reward else None,
        })
    return rows


def _compute_category_metrics(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics_by_group: Dict[str, Dict[str, float]] = {}
    if not rows:
        return metrics_by_group

    unique_indices = _last_indices([row['request_id'] for row in rows])
    unique_rows = [rows[idx] for idx in unique_indices]

    grouped_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in unique_rows:
        reward_value = row['total_reward']
        if reward_value is None:
            continue
        grouped_rewards[row['group_value']][row['prompt_id']].append(float(reward_value))

    for group_value in sorted({row['group_value'] for row in rows}):
        group_rows = [row for row in rows if row['group_value'] == group_value]
        group_unique_rows = [row for row in unique_rows if row['group_value'] == group_value]
        metrics: Dict[str, float] = {}

        total_rewards = [row['total_reward'] for row in group_rows if row['total_reward'] is not None]
        reward_mean = _mean(total_rewards)
        if reward_mean is not None:
            metrics['reward'] = reward_mean

        prompt_groups = grouped_rewards.get(group_value, {})
        if prompt_groups:
            prompt_stds = [_std(values) for values in prompt_groups.values()]
            if prompt_stds:
                metrics['reward_std'] = float(sum(prompt_stds) / len(prompt_stds))
                metrics['frac_reward_zero_std'] = float(sum(std == 0.0 for std in prompt_stds) / len(prompt_stds))

        for reward_name in self.reward_func_names:
            reward_name_values = [
                row['rewards'][reward_name]
                for row in group_unique_rows
                if row['rewards'][reward_name] is not None
            ]
            reward_name_mean = _mean(reward_name_values)
            if reward_name_mean is not None:
                metrics[f'rewards/{reward_name}/mean'] = reward_name_mean
                metrics[f'rewards/{reward_name}/std'] = _std(reward_name_values)

        completion_lengths = [row['completion_length'] for row in group_rows if row['completion_length'] is not None]
        if completion_lengths:
            metrics['completions/mean_length'] = float(sum(completion_lengths) / len(completion_lengths))
            metrics['completions/min_length'] = float(min(completion_lengths))
            metrics['completions/max_length'] = float(max(completion_lengths))

        if group_rows:
            metrics['completions/clipped_ratio'] = float(
                sum(bool(row['truncated']) for row in group_rows) / len(group_rows))

        entropy_values = [row['entropy'] for row in group_rows if row['entropy'] is not None]
        if entropy_values:
            metrics['entropy/mean'] = float(sum(entropy_values) / len(entropy_values))
            metrics['entropy/min'] = float(min(entropy_values))
            metrics['entropy/max'] = float(max(entropy_values))

        num_turns_values = [row['num_turns'] for row in group_unique_rows if row['num_turns'] is not None]
        if num_turns_values:
            metrics['num_turns'] = float(sum(num_turns_values) / len(num_turns_values))

        metrics['samples'] = float(len(group_rows))
        metrics_by_group[group_value] = metrics

    return metrics_by_group


def _patch_grpo_trainer():
    from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer

    if getattr(GRPOTrainer, PLUGIN_TAG, False):
        return

    original_prepare_metrics = GRPOTrainer._prepare_metrics
    original_generate_and_score_completions = GRPOTrainer._generate_and_score_completions
    original_compute_advantages = GRPOTrainer._compute_advantages
    original_update_metrics = GRPOTrainer._update_metrics
    original_log = GRPOTrainer.log

    def _prepare_metrics(self):
        original_prepare_metrics(self)
        group_key = os.environ.get(ENV_KEY, '').strip()
        self._tb_group_key = group_key or None
        self._tb_group_writers = {}
        self._tb_group_summary_writer_cls = None
        self._tb_group_run_root = None
        self._tb_group_windows = {
            'train': _new_group_window(),
            'eval': _new_group_window(),
        }
        if not self._tb_group_key:
            return
        if not self.accelerator.is_main_process:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_group_summary_writer_cls = SummaryWriter
            self._tb_group_run_root = _resolve_group_run_root(self.args.logging_dir)
            logger.info(
                f'Enabled GRPO per-category TensorBoard logging with key `{self._tb_group_key}` '
                f'under `{self._tb_group_run_root}`. Overall metrics remain in `{self.args.logging_dir}`.')
        except Exception as exc:
            logger.warning(f'Failed to initialize TensorBoard SummaryWriter for grouped metrics: {exc}')

    def _generate_and_score_completions(self, inputs):
        result = original_generate_and_score_completions(self, inputs)
        if not getattr(self, '_tb_group_key', None):
            return result

        try:
            window = _get_group_window(self)
            group_values = gather_object([_extract_group_value(inp, self._tb_group_key) for inp in inputs])
            prompt_ids = gather_object([inp.get('prompt_id') for inp in inputs])
            request_ids = gather_object([inp.get('request_id') for inp in inputs])

            local_lengths = [batch['completion_mask'].sum(1).tolist() for batch in result]
            total_lengths = self._gather_and_flatten(
                local_lengths, dtype=torch.float32, device=self.accelerator.device, flatten_level=1).tolist()
            local_truncated = [batch['truncated_mask'].tolist() for batch in result]
            total_truncated = self._gather_and_flatten(
                local_truncated, dtype=torch.bool, device=self.accelerator.device, flatten_level=1).tolist()

            if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in inputs):
                total_num_turns = gather_object([inp['rollout_infos']['num_turns'] for inp in inputs])
            else:
                total_num_turns = [None] * len(group_values)

            window['group_value'].extend(group_values)
            window['prompt_id'].extend(prompt_ids)
            window['request_id'].extend(request_ids)
            window['completion_length'].extend(total_lengths)
            window['truncated'].extend(total_truncated)
            window['num_turns'].extend(total_num_turns)
        except Exception as exc:
            logger.warning(f'Failed to collect grouped GRPO metrics: {exc}')

        return result

    def _compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs):
        advantages = original_compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs)

        if not getattr(self, '_tb_group_key', None):
            return advantages

        try:
            window = _get_group_window(self)
            reward_rows = rewards_per_func.detach().float().cpu().tolist()
            for reward_idx, reward_name in enumerate(self.reward_func_names):
                window['rewards'][reward_name].extend(row[reward_idx] for row in reward_rows)
        except Exception as exc:
            logger.warning(f'Failed to collect grouped GRPO reward metrics: {exc}')

        return advantages

    def _update_metrics(self, metrics_data):
        if getattr(self, '_tb_group_key', None):
            try:
                entropy_metrics = metrics_data.get('entropy') or {}
                entropy_logs = entropy_metrics.get('entropy_logs')
                if entropy_logs is not None:
                    window = _get_group_window(self, metrics_data.get('mode'))
                    window['entropy'].extend(entropy_logs)
            except Exception as exc:
                logger.warning(f'Failed to collect grouped GRPO entropy metrics: {exc}')

        return original_update_metrics(self, metrics_data)

    def log(self, logs, start_time=None):
        original_log(self, logs, start_time)

        if not getattr(self, '_tb_group_key', None):
            return
        if not self.accelerator.is_main_process or self._tb_group_summary_writer_cls is None:
            return

        mode = 'train' if self.model.training else 'eval'
        try:
            window = _get_group_window(self, mode)
            rows = _build_rows(self, window)
            metrics_by_group = _compute_category_metrics(self, rows)

            for group_value, metric_dict in metrics_by_group.items():
                writer = _get_group_writer(self, group_value)
                for metric_name, metric_value in metric_dict.items():
                    writer.add_scalar(f'{mode}/{metric_name}', metric_value, self.state.global_step)
            for writer in self._tb_group_writers.values():
                writer.flush()
        except Exception as exc:
            logger.warning(f'Failed to write grouped GRPO metrics to TensorBoard: {exc}')
        finally:
            _clear_group_window(_get_group_window(self, mode))

    GRPOTrainer._prepare_metrics = _prepare_metrics
    GRPOTrainer._generate_and_score_completions = _generate_and_score_completions
    GRPOTrainer._compute_advantages = _compute_advantages
    GRPOTrainer._update_metrics = _update_metrics
    GRPOTrainer.log = log
    setattr(GRPOTrainer, PLUGIN_TAG, True)


_patch_grpo_trainer()
