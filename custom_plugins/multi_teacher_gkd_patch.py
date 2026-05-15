from contextlib import nullcontext
from functools import wraps

_PATCH_APPLIED = False
_PATCH_CONFIG = {
    'teacher_models': None,
    'teacher_model_types': None,
    'teacher_model_revisions': None,
    'teacher_weights': None,
    'teacher_seq_kd_index': 0,
}


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _broadcast_optional_list(value, length, name):
    value = _as_list(value)
    if value is None:
        return [None] * length
    if len(value) == 1 and length > 1:
        return value * length
    if len(value) != length:
        raise ValueError(f'Length of {name} ({len(value)}) must match teacher_models ({length}).')
    return value


def _normalize_weights(weights, length):
    if weights is None:
        return [1.0 / length] * length
    weights = [float(weight) for weight in weights]
    if len(weights) != length:
        raise ValueError(f'Length of teacher_weights ({len(weights)}) must match teacher_models ({length}).')
    if any(weight < 0 for weight in weights):
        raise ValueError('teacher_weights must be non-negative.')
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError('At least one teacher weight must be positive.')
    return [weight / weight_sum for weight in weights]


def _validate_config():
    teacher_models = _as_list(_PATCH_CONFIG['teacher_models'])
    if not teacher_models:
        return None

    teacher_model_types = _broadcast_optional_list(_PATCH_CONFIG['teacher_model_types'], len(teacher_models),
                                                   'teacher_model_types')
    teacher_model_revisions = _broadcast_optional_list(_PATCH_CONFIG['teacher_model_revisions'], len(teacher_models),
                                                       'teacher_model_revisions')
    teacher_weights = _normalize_weights(_PATCH_CONFIG['teacher_weights'], len(teacher_models))
    teacher_seq_kd_index = int(_PATCH_CONFIG['teacher_seq_kd_index'])
    if not 0 <= teacher_seq_kd_index < len(teacher_models):
        raise ValueError(
            f'teacher_seq_kd_index must be in [0, {len(teacher_models) - 1}], got {teacher_seq_kd_index}.')
    return teacher_models, teacher_model_types, teacher_model_revisions, teacher_weights, teacher_seq_kd_index


def apply_patch(
    *,
    teacher_models=None,
    teacher_model_types=None,
    teacher_model_revisions=None,
    teacher_weights=None,
    teacher_seq_kd_index=0,
):
    global _PATCH_APPLIED

    _PATCH_CONFIG['teacher_models'] = teacher_models
    _PATCH_CONFIG['teacher_model_types'] = teacher_model_types
    _PATCH_CONFIG['teacher_model_revisions'] = teacher_model_revisions
    _PATCH_CONFIG['teacher_weights'] = teacher_weights
    _PATCH_CONFIG['teacher_seq_kd_index'] = teacher_seq_kd_index
    resolved = _validate_config()
    if resolved is None:
        return
    if _PATCH_APPLIED:
        return

    import torch
    import torch.nn.functional as F

    from swift.llm import disable_gradient_checkpointing
    from swift.llm.train.rlhf import SwiftRLHF
    from swift.trainers.rlhf_trainer.gkd_trainer import DataSource, GKDTrainer
    from swift.trainers.rlhf_trainer.utils import prepare_deepspeed
    from swift.utils import get_logger

    logger = get_logger()
    original_prepare_model_tokenizer = SwiftRLHF._prepare_model_tokenizer
    original_gkd_init = GKDTrainer.__init__
    original_compute_loss = GKDTrainer.compute_loss

    def _get_config():
        config = _validate_config()
        if config is None:
            return None
        return config

    @wraps(original_prepare_model_tokenizer)
    def patched_prepare_model_tokenizer(self):
        config = _get_config()
        if config is None or self.args.rlhf_type != 'gkd':
            return original_prepare_model_tokenizer(self)

        teacher_models, teacher_types, teacher_revisions, teacher_weights, teacher_seq_kd_index = config
        args = self.args
        origin_teacher_model = args.teacher_model
        origin_teacher_model_type = args.teacher_model_type
        origin_teacher_model_revision = args.teacher_model_revision
        origin_reward_adapters = args.reward_adapters

        args.teacher_model = teacher_models[0]
        args.teacher_model_type = teacher_types[0]
        args.teacher_model_revision = teacher_revisions[0]
        args.reward_adapters = args.teacher_adapters
        try:
            original_prepare_model_tokenizer(self)
            prepared_teacher_models = [self.teacher_model]
            for teacher_model, teacher_type, teacher_revision in zip(
                    teacher_models[1:], teacher_types[1:], teacher_revisions[1:]):
                args.teacher_model = teacher_model
                result = self._prepare_single_model('teacher', 'teacher', teacher_type, teacher_revision)
                if result is not None:
                    teacher, _ = result
                    prepared_teacher_models.append(teacher)
        finally:
            args.reward_adapters = origin_reward_adapters
            args.teacher_model = teacher_models
            args.teacher_model_type = teacher_types
            args.teacher_model_revision = teacher_revisions

        self.teacher_model = prepared_teacher_models
        args._multi_teacher_gkd_weights = teacher_weights
        args._multi_teacher_gkd_seq_kd_index = teacher_seq_kd_index
        args._multi_teacher_gkd_models = teacher_models
        args.teacher_model = teacher_models
        args.teacher_model_type = teacher_types
        args.teacher_model_revision = teacher_revisions
        if origin_teacher_model and origin_teacher_model != teacher_models[0]:
            logger.warning('Ignoring original teacher_model=%s in favor of multi-teacher wrapper config.',
                           origin_teacher_model)
        if origin_teacher_model_type and origin_teacher_model_type != teacher_types[0]:
            logger.warning('Ignoring original teacher_model_type=%s in favor of multi-teacher wrapper config.',
                           origin_teacher_model_type)
        if origin_teacher_model_revision and origin_teacher_model_revision != teacher_revisions[0]:
            logger.warning('Ignoring original teacher_model_revision=%s in favor of multi-teacher wrapper config.',
                           origin_teacher_model_revision)

    def _prepare_extra_teacher(self, teacher_model, teacher_deepspeed_config):
        if self.is_deepspeed_enabled:
            if teacher_deepspeed_config is not None:
                return prepare_deepspeed(
                    teacher_model,
                    self.accelerator,
                    deepspeed_config=teacher_deepspeed_config,
                    training_args=self.args)
            return prepare_deepspeed(teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            from swift.trainers.rlhf_trainer.utils import prepare_fsdp
            return prepare_fsdp(teacher_model, self.accelerator)
        else:
            return self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

    @wraps(original_gkd_init)
    def patched_gkd_init(self, model=None, *_args, **kwargs):
        teacher_model = kwargs.get('teacher_model')
        if not isinstance(teacher_model, (list, tuple)):
            original_gkd_init(self, model, *_args, **kwargs)
            self.teacher_models = [self.teacher_model]
            self.teacher_model_weights = [1.0]
            self.teacher_seq_kd_index = 0
            return

        teacher_models = list(teacher_model)
        if not teacher_models:
            raise ValueError('Multi-teacher GKD requires at least one teacher model.')

        args = kwargs['args']
        teacher_deepspeed_config = kwargs.get('teacher_deepspeed_config', None)
        teacher_weights = getattr(args, '_multi_teacher_gkd_weights', None) or [1.0 / len(teacher_models)] * len(
            teacher_models)
        teacher_seq_kd_index = getattr(args, '_multi_teacher_gkd_seq_kd_index', 0)

        kwargs['teacher_model'] = teacher_models[0]
        original_gkd_init(self, model, *_args, **kwargs)
        self.teacher_models = [self.teacher_model]
        self.teacher_model_weights = teacher_weights
        self.teacher_seq_kd_index = teacher_seq_kd_index
        self._multi_teacher_vocab_warning_logged = False

        for teacher in teacher_models[1:]:
            prepared_teacher = _prepare_extra_teacher(self, teacher, teacher_deepspeed_config)
            prepared_teacher.eval()
            if self.args.offload_teacher_model:
                self.offload_model(self.accelerator.unwrap_model(prepared_teacher))
            self.teacher_models.append(prepared_teacher)

        self.teacher_model = self.teacher_models[self.teacher_seq_kd_index]

    def _load_teacher_model_context(self, teacher_model):
        if not self.args.offload_teacher_model:
            return nullcontext()

        class _LoadTeacherContext:

            def __enter__(_self):
                self.load_model(self.accelerator.unwrap_model(teacher_model))

            def __exit__(_self, exc_type, exc, tb):
                self.offload_model(self.accelerator.unwrap_model(teacher_model))

        return _LoadTeacherContext()

    def _match_teacher_logits_to_student_vocab(self, teacher_logits, student_vocab_size):
        teacher_vocab_size = teacher_logits.shape[-1]
        if teacher_vocab_size == student_vocab_size:
            return teacher_logits

        if not getattr(self, '_multi_teacher_vocab_warning_logged', False):
            logger.warning(
                'Multi-teacher GKD works best when all teachers and the student share the same tokenizer/vocab. '
                'Current teacher logits will be truncated or padded to the student vocab size.')
            self._multi_teacher_vocab_warning_logged = True

        if teacher_vocab_size > student_vocab_size:
            return teacher_logits[..., :student_vocab_size]

        pad_size = student_vocab_size - teacher_vocab_size
        pad_value = torch.finfo(teacher_logits.dtype).min
        return F.pad(teacher_logits, (0, pad_size), 'constant', pad_value)

    def _get_multi_teacher_log_probs(self, model_inputs, mask, student_vocab_size):
        ensemble_log_probs = None
        for teacher, weight in zip(self.teacher_models, self.teacher_model_weights):
            load_context = _load_teacher_model_context(self, teacher)
            with torch.no_grad(), load_context, disable_gradient_checkpointing(
                    teacher, self.args.gradient_checkpointing_kwargs):
                outputs_teacher = teacher(**model_inputs)

            shifted_teacher_logits = outputs_teacher.logits[mask][None]
            shifted_teacher_logits = _match_teacher_logits_to_student_vocab(
                self, shifted_teacher_logits, student_vocab_size)
            teacher_log_probs = F.log_softmax(shifted_teacher_logits / self.temperature, dim=-1)
            weighted_teacher_log_probs = teacher_log_probs + teacher_log_probs.new_tensor(weight).log()
            if ensemble_log_probs is None:
                ensemble_log_probs = weighted_teacher_log_probs
            else:
                ensemble_log_probs = torch.logaddexp(ensemble_log_probs, weighted_teacher_log_probs)
            del outputs_teacher, shifted_teacher_logits, teacher_log_probs, weighted_teacher_log_probs
        return ensemble_log_probs

    @staticmethod
    def generalized_jsd_loss_with_teacher_log_probs(
        student_logits,
        teacher_log_probs,
        labels=None,
        beta=0.5,
        temperature=1.0,
        chunk_size=512,
    ):
        student_logits = student_logits / temperature

        if labels is not None:
            mask = labels != -100
            student_logits = student_logits[mask]
            teacher_log_probs = teacher_log_probs[mask]
            num_valid = mask.sum()
        else:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_log_probs = teacher_log_probs.view(-1, teacher_log_probs.size(-1))
            num_valid = student_logits.size(0)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits[start_idx:end_idx]
            t_log_probs = teacher_log_probs[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            del s_chunk

            if beta == 0:
                jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
            elif beta == 1:
                jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )
                kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
                kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
                del mixture_log_probs

                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        return total_loss / num_valid

    @wraps(original_compute_loss)
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_models = getattr(self, 'teacher_models', None)
        if not teacher_models or len(teacher_models) == 1:
            return original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
        if getattr(self, 'use_liger_gkd_loss', False):
            raise NotImplementedError('Multi-teacher GKD does not support `use_liger_kernel` yet.')

        data_source = inputs.pop('_data_source', DataSource.DATASET)
        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']

        if self.args.sft_alpha > 0:
            model_inputs['labels'] = inputs['labels']
        outputs_student = model(**model_inputs)
        model_inputs.pop('labels', None)

        shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
        mask = shifted_labels != -100
        shifted_student_logits = outputs_student.logits[mask][None]
        teacher_log_probs = _get_multi_teacher_log_probs(
            self,
            model_inputs=model_inputs,
            mask=mask,
            student_vocab_size=shifted_student_logits.shape[-1],
        )
        loss = generalized_jsd_loss_with_teacher_log_probs(
            student_logits=shifted_student_logits,
            teacher_log_probs=teacher_log_probs,
            beta=self.beta,
            temperature=self.temperature,
        )
        if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
            loss = loss + self.args.sft_alpha * outputs_student.loss

        if return_outputs:
            return loss, outputs_student
        return loss

    SwiftRLHF._prepare_model_tokenizer = patched_prepare_model_tokenizer
    GKDTrainer.__init__ = patched_gkd_init
    GKDTrainer.compute_loss = patched_compute_loss
    GKDTrainer.generalized_jsd_loss_with_teacher_log_probs = generalized_jsd_loss_with_teacher_log_probs
    GKDTrainer._patched_multi_teacher_gkd_apply_patch = True
    _PATCH_APPLIED = True
