import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import wraps

_PATCH_APPLIED = False
_PATCH_DEFAULTS = {
    'encode_num_workers': None,
    'infer_prefetch_size': None,
}


def _validate_positive_or_none(name, value):
    if value is not None and value <= 0:
        raise ValueError(f'{name} must be greater than 0, got {value}.')


def apply_patch(*, encode_num_workers=None, infer_prefetch_size=None):
    global _PATCH_APPLIED

    _validate_positive_or_none('encode_num_workers', encode_num_workers)
    _validate_positive_or_none('infer_prefetch_size', infer_prefetch_size)
    _PATCH_DEFAULTS['encode_num_workers'] = encode_num_workers
    _PATCH_DEFAULTS['infer_prefetch_size'] = infer_prefetch_size
    if _PATCH_APPLIED:
        return

    import torch
    from tqdm import tqdm

    from swift.llm.infer.infer_engine.infer_engine import InferEngine
    from swift.llm.infer.infer_engine.vllm_engine import VllmEngine
    from swift.llm.infer.protocol import RequestConfig
    from swift.utils import get_dist_setting, is_dist

    original_init = VllmEngine.__init__
    original_infer = VllmEngine.infer

    def _resolve_encode_num_workers(self, num_requests):
        configured = getattr(self, '_patched_vllm_encode_num_workers', None)
        if configured is not None:
            return configured
        cpu_count = os.cpu_count() or 1
        return max(1, min(32, cpu_count, max(num_requests, 1)))

    def _resolve_infer_prefetch_size(self, num_requests, encode_num_workers):
        configured = getattr(self, '_patched_vllm_infer_prefetch_size', None)
        if configured is not None:
            prefetch_size = configured
        else:
            max_num_seqs = getattr(getattr(self, 'engine_args', None), 'max_num_seqs', None) or 256
            prefetch_size = max(max_num_seqs, 2 * encode_num_workers)
        return max(1, min(prefetch_size, max(num_requests, 1)))

    def _ensure_encode_executor(self, num_requests=1):
        resolved_workers = _resolve_encode_num_workers(self, num_requests)
        executor = getattr(self, '_patched_vllm_encode_executor', None)
        current_workers = getattr(self, '_patched_vllm_encode_executor_max_workers', None)
        if executor is None or current_workers != resolved_workers:
            if executor is not None:
                executor.shutdown(wait=True)
            executor = ThreadPoolExecutor(max_workers=resolved_workers)
            self._patched_vllm_encode_executor = executor
            self._patched_vllm_encode_executor_max_workers = resolved_workers
        return resolved_workers

    def _run_coro_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return InferEngine.safe_asyncio_run(coro)

    def _prepare_async_batch_infer(self):
        if hasattr(self.engine, 'engine'):
            self.engine.engine.model_executor.parallel_worker_tasks = None

    async def _infer_async_batch(self, infer_requests, request_config, template, adapter_request, metrics, prog_bar,
                                 prefetch_size):
        if not infer_requests:
            return []

        outputs = [None] * len(infer_requests)
        pending_tasks = {}
        next_index = 0

        def _create_task(index):
            task = asyncio.create_task(
                self.infer_async(
                    infer_requests[index],
                    request_config,
                    template=template,
                    adapter_request=adapter_request,
                ))
            pending_tasks[task] = index

        while next_index < min(prefetch_size, len(infer_requests)):
            _create_task(next_index)
            next_index += 1

        while pending_tasks:
            done, _ = await asyncio.wait(tuple(pending_tasks), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = pending_tasks.pop(task)
                try:
                    outputs[index] = task.result()
                except Exception as e:
                    if getattr(self, 'strict', True):
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        if pending_tasks:
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                        raise
                    outputs[index] = e
                prog_bar.update()
                self._update_metrics(outputs[index], metrics)
                if next_index < len(infer_requests):
                    _create_task(next_index)
                    next_index += 1
        return outputs

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._patched_vllm_encode_num_workers = _PATCH_DEFAULTS['encode_num_workers']
        self._patched_vllm_infer_prefetch_size = _PATCH_DEFAULTS['infer_prefetch_size']
        self._patched_vllm_encode_executor = None
        self._patched_vllm_encode_executor_max_workers = None

    @wraps(VllmEngine.infer_async)
    async def patched_infer_async(
        self,
        infer_request,
        request_config=None,
        *,
        template=None,
        adapter_request=None,
        pre_infer_hook=None,
    ):
        if not self.use_async_engine:
            raise ValueError('If you want to use `infer_async`, you need to pass `use_async_engine` as True.')
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template

        template.set_mode('vllm')
        loop = asyncio.get_running_loop()
        _ensure_encode_executor(self)
        with torch.inference_mode():
            inputs = await loop.run_in_executor(
                self._patched_vllm_encode_executor,
                template.encode,
                infer_request,
                True,
            )
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template.template_meta)
        kwargs = {
            'template': template,
            'inputs': inputs,
            'generation_config': generation_config,
            'adapter_request': adapter_request,
            'request_config': request_config,
        }
        if hasattr(infer_request, 'uuid') and infer_request.uuid:
            kwargs['request_id'] = infer_request.uuid
        if pre_infer_hook:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:
            return self._infer_stream_async(**kwargs)
        return await self._infer_full_async(**kwargs)

    @wraps(original_infer)
    def patched_infer(
        self,
        infer_requests,
        request_config=None,
        metrics=None,
        *,
        template=None,
        use_tqdm=None,
        adapter_request=None,
    ):
        if not self.use_async_engine:
            return original_infer(
                self,
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request,
            )

        request_config = deepcopy(request_config or RequestConfig())
        if request_config.stream:
            return original_infer(
                self,
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request,
            )

        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        rank = get_dist_setting()[0]
        if is_dist() and rank % self.engine_args.tensor_parallel_size != 0:
            use_tqdm = False
        if template is None:
            template = self.default_template
        template.set_mode('vllm')
        _prepare_async_batch_infer(self)
        resolved_encode_num_workers = _ensure_encode_executor(self, len(infer_requests))
        prefetch_size = _resolve_infer_prefetch_size(self, len(infer_requests), resolved_encode_num_workers)
        prog_bar = tqdm(total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm)
        try:
            return _run_coro_sync(
                _infer_async_batch(
                    self,
                    infer_requests,
                    request_config,
                    template,
                    adapter_request,
                    metrics,
                    prog_bar,
                    prefetch_size,
                ))
        finally:
            prog_bar.close()

    VllmEngine.__init__ = patched_init
    VllmEngine.infer_async = patched_infer_async
    VllmEngine.infer = patched_infer
    VllmEngine._patched_vllm_apply_patch = True
    _PATCH_APPLIED = True
