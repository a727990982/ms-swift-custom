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
    if value != value:
        return None
    return value


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    clean_values = [float(v) for v in values if v is not None]
    if not clean_values:
        return None
    return float(sum(clean_values) / len(clean_values))


def _std_or_zero(values: List[Optional[float]]) -> float:
    clean_values = [float(v) for v in values if v is not None]
    if len(clean_values) <= 1:
        return 0.0
    return _std(clean_values)


def _new_group_window() -> Dict[str, Any]:
    return {'metric_values': defaultdict(lambda: defaultdict(list))}


def _clear_group_window(window: Dict[str, Any]) -> None:
    window['metric_values'] = defaultdict(lambda: defaultdict(list))


def _get_group_window(self, mode: Optional[str] = None) -> Dict[str, Any]:
    if mode is None:
        mode = 'train' if self.model.training else 'eval'
    return self._tb_group_windows[mode]


def _append_group_metric(window: Dict[str, Any], group_value: str, metric_name: str, metric_value: float) -> None:
    window['metric_values'][group_value][metric_name].append(float(metric_value))


def _build_window_metrics(window: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    metrics_by_group: Dict[str, Dict[str, float]] = {}
    for group_value, metric_values in window['metric_values'].items():
        metrics: Dict[str, float] = {}
        for metric_name, values in metric_values.items():
            if not values:
                continue
            if metric_name == 'samples':
                metrics[metric_name] = float(sum(values))
            else:
                metrics[metric_name] = float(sum(values) / len(values))
        if metrics:
            metrics_by_group[group_value] = metrics
    return metrics_by_group


def _group_run_name(group_key: str, group_value: str) -> str:
    return f'{_sanitize_tag(group_key)}={_sanitize_tag(group_value)}'


def _resolve_group_run_root(logging_dir: str) -> Path:
    logging_dir = Path(logging_dir)
    if logging_dir.name == 'overall':
        return logging_dir.parent / 'grouped'
    return logging_dir / 'grouped'


def _get_group_writer(self, group_value: str):
    writer = self._tb_group_writers.get(group_value)
    if writer is not None:
        return writer

    log_dir = self._tb_group_run_root / _group_run_name(self._tb_group_key, group_value)
    writer = self._tb_group_summary_writer_cls(log_dir=str(log_dir))
    self._tb_group_writers[group_value] = writer
    return writer


def _gather_list(values: List[Any]) -> List[Any]:
    gathered = gather_object(values)
    if isinstance(gathered, tuple):
        return list(gathered)
    if isinstance(gathered, list):
        if gathered and all(isinstance(item, (list, tuple)) for item in gathered):
            flat = []
            for item in gathered:
                flat.extend(list(item))
            return flat
        return gathered
    return [gathered]


def _collect_global_group_fields(self, inputs) -> Dict[str, List[Any]]:
    global_inputs = _gather_list(inputs)
    return {
        'group_values': [_extract_group_value(inp, self._tb_group_key) for inp in global_inputs],
        'prompt_ids': [inp.get('prompt_id') for inp in global_inputs],
        'request_ids': [inp.get('request_id') for inp in global_inputs],
    }


def _record_generation_metrics(self, window: Dict[str, Any], inputs, batch_encoded_inputs) -> None:
    gas_chunks = self.split_by_mini_batches(inputs)
    if len(gas_chunks) != len(batch_encoded_inputs):
        logger.warning('Skipped grouped generation metrics because chunk count did not match encoded batches.')
        return

    local_rows = []
    for batch_inputs, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
        batch_size = len(batch_inputs)
        batch_group_values = [_extract_group_value(inp, self._tb_group_key) for inp in batch_inputs]
        batch_request_ids = [inp.get('request_id') for inp in batch_inputs]
        if all('rollout_infos' in inp and 'num_turns' in inp['rollout_infos'] for inp in batch_inputs):
            batch_num_turns = [inp['rollout_infos']['num_turns'] for inp in batch_inputs]
        else:
            batch_num_turns = [None] * batch_size
        if len(batch_group_values) != batch_size or len(batch_request_ids) != batch_size or len(batch_num_turns) != batch_size:
            logger.warning('Skipped grouped generation metrics because local group metadata did not match batch size.')
            return
        batch_lengths = batch_encoded['completion_mask'].sum(1).tolist()
        batch_truncated = batch_encoded['truncated_mask'].tolist()
        if len(batch_lengths) != batch_size or len(batch_truncated) != batch_size:
            logger.warning('Skipped grouped generation metrics because encoded batch tensors did not match batch size.')
            return
        batch_encoded['tb_group_values'] = batch_group_values
        for idx in range(batch_size):
            local_rows.append({
                'group_value': batch_group_values[idx],
                'request_id': batch_request_ids[idx],
                'length': float(batch_lengths[idx]),
                'truncated': bool(batch_truncated[idx]),
                'num_turns': batch_num_turns[idx],
            })

    rows = _gather_list(local_rows)
    group_to_rows = defaultdict(list)
    for row in rows:
        group_to_rows[row['group_value']].append(row)

    last_indices = _last_indices([row['request_id'] for row in rows]) if self.dynamic_num_samples else None
    for group_value, group_rows in group_to_rows.items():
        group_lengths = [row['length'] for row in group_rows]
        _append_group_metric(window, group_value, 'samples', float(len(group_rows)))
        _append_group_metric(window, group_value, 'completions/mean_length',
                             float(sum(group_lengths) / len(group_lengths)))
        _append_group_metric(window, group_value, 'completions/min_length', float(min(group_lengths)))
        _append_group_metric(window, group_value, 'completions/max_length', float(max(group_lengths)))

        if not self.dynamic_num_samples:
            _append_group_metric(window, group_value, 'completions/clipped_ratio',
                                 float(sum(row['truncated'] for row in group_rows) / len(group_rows)))
            group_num_turns = [float(row['num_turns']) for row in group_rows if row['num_turns'] is not None]
            if len(group_num_turns) == len(group_rows):
                _append_group_metric(window, group_value, 'num_turns',
                                     float(sum(group_num_turns) / len(group_num_turns)))
        else:
            final_rows = [rows[idx] for idx in last_indices if rows[idx]['group_value'] == group_value]
            if final_rows:
                _append_group_metric(window, group_value, 'completions/clipped_ratio',
                                     float(sum(row['truncated'] for row in final_rows) / len(final_rows)))
                final_num_turns = [float(row['num_turns']) for row in final_rows if row['num_turns'] is not None]
                if len(final_num_turns) == len(final_rows):
                    _append_group_metric(window, group_value, 'num_turns',
                                         float(sum(final_num_turns) / len(final_num_turns)))


def _record_reward_metrics(self, window: Dict[str, Any], inputs, rewards_per_func, batch_encoded_inputs) -> None:
    fields = _collect_global_group_fields(self, inputs)
    reward_rows = rewards_per_func.detach().float().cpu().tolist()
    weights = self.reward_weights.detach().float().cpu().tolist()
    total_rewards = []
    for row in reward_rows:
        reward_total = 0.0
        for reward_value, weight in zip(row, weights):
            safe_value = _safe_float(reward_value)
            if safe_value is not None:
                reward_total += safe_value * float(weight)
        total_rewards.append(reward_total)

    if self.kl_in_reward and self.beta != 0.0:
        local_kl_values = []
        for batch_encoded in batch_encoded_inputs:
            old_per_token_logps = batch_encoded['old_per_token_logps']
            ref_per_token_logps = batch_encoded['ref_per_token_logps']
            completion_mask = batch_encoded['completion_mask']
            local_kl_values.extend((((old_per_token_logps - ref_per_token_logps) * completion_mask).sum(-1)
                                    ).detach().float().cpu().tolist())
        kl_values = _gather_list(local_kl_values)
        if len(kl_values) != len(total_rewards):
            logger.warning('Skipped grouped reward metrics because global KL values and rewards had different lengths.')
            return
        total_rewards = [
            float(reward_value - self.beta * kl_value)
            for reward_value, kl_value in zip(total_rewards, kl_values)
        ]

    total_count = min(len(fields['group_values']), len(total_rewards), len(reward_rows))
    if total_count == 0:
        return
    if len(fields['group_values']) != len(total_rewards) or len(total_rewards) != len(reward_rows):
        logger.warning('Skipped grouped reward metrics because global group fields and rewards had different lengths.')
        return

    rows = [{
        'group_value': fields['group_values'][idx],
        'prompt_id': fields['prompt_ids'][idx],
        'request_id': fields['request_ids'][idx],
        'total_reward': float(total_rewards[idx]),
        'reward_values': reward_rows[idx],
    } for idx in range(total_count)]
    if self.dynamic_num_samples:
        rows = [rows[idx] for idx in _last_indices([row['request_id'] for row in rows])]

    group_to_prompt_rewards = defaultdict(lambda: defaultdict(list))
    group_to_rows = defaultdict(list)
    for row in rows:
        group_to_prompt_rewards[row['group_value']][row['prompt_id']].append(row['total_reward'])
        group_to_rows[row['group_value']].append(row)

    for group_value, prompt_rewards in group_to_prompt_rewards.items():
        prompt_means = [_mean(values) for values in prompt_rewards.values()]
        reward_mean = _mean_or_none(prompt_means)
        if reward_mean is not None:
            _append_group_metric(window, group_value, 'reward', reward_mean)

        prompt_stds = [_std(values) if len(values) > 1 else 0.0 for values in prompt_rewards.values()]
        if self.scale_rewards in ['group', 'none']:
            reward_std = _mean_or_none(prompt_stds)
        else:
            reward_std = _std_or_zero([row['total_reward'] for row in group_to_rows[group_value]])
        if reward_std is not None:
            _append_group_metric(window, group_value, 'reward_std', reward_std)
        if prompt_stds:
            _append_group_metric(window, group_value, 'frac_reward_zero_std',
                                 float(sum(std == 0.0 for std in prompt_stds) / len(prompt_stds)))

        for reward_idx, reward_name in enumerate(self.reward_func_names):
            reward_name_values = [_safe_float(row['reward_values'][reward_idx]) for row in group_to_rows[group_value]]
            reward_name_mean = _mean_or_none(reward_name_values)
            if reward_name_mean is not None:
                _append_group_metric(window, group_value, f'rewards/{reward_name}/mean', reward_name_mean)
                _append_group_metric(window, group_value, f'rewards/{reward_name}/std',
                                     _std_or_zero(reward_name_values))


def _record_entropy_metrics(self, window: Dict[str, Any], inputs, metrics_data) -> None:
    entropy_metrics = metrics_data.get('entropy') or {}
    entropy_logs = entropy_metrics.get('entropy_logs')
    group_values = inputs.get('tb_group_values')
    if entropy_logs is None or group_values is None:
        return

    flattened_group_values = _gather_list(group_values)
    if len(flattened_group_values) != len(entropy_logs):
        logger.warning('Skipped grouped entropy metrics because group values and entropy logs had different lengths.')
        return

    grouped_entropies = defaultdict(list)
    for group_value, entropy_value in zip(flattened_group_values, entropy_logs):
        safe_entropy = _safe_float(entropy_value)
        if safe_entropy is not None:
            grouped_entropies[group_value].append(safe_entropy)

    for group_value, entropy_values in grouped_entropies.items():
        _append_group_metric(window, group_value, 'entropy/mean', float(sum(entropy_values) / len(entropy_values)))
        _append_group_metric(window, group_value, 'entropy/min', float(min(entropy_values)))
        _append_group_metric(window, group_value, 'entropy/max', float(max(entropy_values)))


def _patch_grpo_trainer():
    from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer

    if getattr(GRPOTrainer, PLUGIN_TAG, False):
        return

    original_prepare_metrics = GRPOTrainer._prepare_metrics
    original_prepare_batch_inputs = GRPOTrainer._prepare_batch_inputs
    original_compute_advantages = GRPOTrainer._compute_advantages
    original_compute_loss_and_metrics = GRPOTrainer._compute_loss_and_metrics
    original_get_chunked_inputs = GRPOTrainer.get_chunked_inputs
    original_log = GRPOTrainer.log

    def _prepare_metrics(self):
        original_prepare_metrics(self)
        self._tb_group_key = os.environ.get(ENV_KEY, '').strip() or None
        self._tb_group_writers = {}
        self._tb_group_summary_writer_cls = None
        self._tb_group_run_root = None
        self._tb_group_windows = {'train': _new_group_window(), 'eval': _new_group_window()}
        if not self._tb_group_key or not self.accelerator.is_main_process:
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

    def _prepare_batch_inputs(self, inputs):
        result = original_prepare_batch_inputs(self, inputs)
        if not getattr(self, '_tb_group_key', None):
            return result
        try:
            window = _get_group_window(self)
            _record_generation_metrics(self, window, inputs, result)
        except Exception as exc:
            logger.warning(f'Failed to collect grouped GRPO generation metrics: {exc}')
        return result

    def _compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs):
        advantages = original_compute_advantages(self, inputs, rewards_per_func, batch_encoded_inputs)
        if not getattr(self, '_tb_group_key', None):
            return advantages
        try:
            _record_reward_metrics(self, _get_group_window(self), inputs, rewards_per_func, batch_encoded_inputs)
        except Exception as exc:
            logger.warning(f'Failed to collect grouped GRPO reward metrics: {exc}')
        return advantages

    def _compute_loss_and_metrics(self, model, inputs):
        loss, metrics_data = original_compute_loss_and_metrics(self, model, inputs)
        if not getattr(self, '_tb_group_key', None):
            return loss, metrics_data
        try:
            _record_entropy_metrics(self, _get_group_window(self, metrics_data.get('mode')), inputs, metrics_data)
        except Exception as exc:
            logger.warning(f'Failed to collect grouped GRPO entropy metrics: {exc}')
        return loss, metrics_data

    def get_chunked_inputs(self, inputs, start_idx, end_idx):
        chunk_inputs = original_get_chunked_inputs(self, inputs, start_idx, end_idx)
        if 'tb_group_values' in chunk_inputs and isinstance(chunk_inputs['tb_group_values'], (list, tuple)):
            chunk_inputs['tb_group_values'] = list(chunk_inputs['tb_group_values'][start_idx:end_idx])
        return chunk_inputs

    def log(self, logs, start_time=None):
        original_log(self, logs, start_time)
        if not getattr(self, '_tb_group_key', None):
            return
        if not self.accelerator.is_main_process or self._tb_group_summary_writer_cls is None:
            return

        mode = 'train' if self.model.training else 'eval'
        window = _get_group_window(self, mode)
        try:
            for group_value, metric_dict in _build_window_metrics(window).items():
                writer = _get_group_writer(self, group_value)
                for metric_name, metric_value in metric_dict.items():
                    writer.add_scalar(f'{mode}/{metric_name}', metric_value, self.state.global_step)
            for writer in self._tb_group_writers.values():
                writer.flush()
        except Exception as exc:
            logger.warning(f'Failed to write grouped GRPO metrics to TensorBoard: {exc}')
        finally:
            _clear_group_window(window)

    GRPOTrainer._prepare_metrics = _prepare_metrics
    GRPOTrainer._prepare_batch_inputs = _prepare_batch_inputs
    GRPOTrainer._compute_advantages = _compute_advantages
    GRPOTrainer._compute_loss_and_metrics = _compute_loss_and_metrics
    GRPOTrainer.get_chunked_inputs = get_chunked_inputs
    GRPOTrainer.log = log
    setattr(GRPOTrainer, PLUGIN_TAG, True)


_patch_grpo_trainer()
