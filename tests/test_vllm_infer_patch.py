import contextlib
import importlib
import sys
import types
import unittest
from types import MethodType, SimpleNamespace

from scripts.run_vllm_infer_patched import _has_explicit_option, _parse_wrapper_args


class _FakeMetric:

    def __init__(self):
        self.responses = []

    def update(self, response):
        self.responses.append(response)


class _FakeTemplate:

    def __init__(self):
        self.mode = None

    def set_mode(self, mode):
        self.mode = mode


class _FakeTqdm:

    def __init__(self, total=0, dynamic_ncols=True, disable=False):
        self.total = total
        self.disable = disable
        self.updates = 0
        self.closed = False

    def update(self, n=1):
        self.updates += n

    def close(self):
        self.closed = True


class _FakeRequestConfig:

    def __init__(self, stream=False):
        self.stream = stream


class _FakeInferEngine:

    @staticmethod
    def safe_asyncio_run(coro):
        import asyncio
        return asyncio.run(coro)


class _FakeVllmEngine:

    def __init__(self, *args, use_async_engine=False, **kwargs):
        self.use_async_engine = use_async_engine
        self.engine_args = SimpleNamespace(tensor_parallel_size=1, max_num_seqs=4)
        self.default_template = _FakeTemplate()
        self.engine = SimpleNamespace(
            engine=SimpleNamespace(model_executor=SimpleNamespace(parallel_worker_tasks='sentinel')))
        self.strict = True

    def infer(self, *args, **kwargs):
        return 'original-infer'

    async def infer_async(self, *args, **kwargs):
        return 'original-infer-async'

    def _update_metrics(self, result, metrics=None):
        if metrics is None:
            return result
        result_origin = result
        if not isinstance(result, (list, tuple)):
            result = [result]
        for response in result:
            if response is None or isinstance(response, Exception):
                continue
            for metric in metrics:
                metric.update(response)
        return result_origin


@contextlib.contextmanager
def _stubbed_patch_env():
    saved_modules = dict(sys.modules)
    try:
        swift_pkg = types.ModuleType('swift')
        llm_pkg = types.ModuleType('swift.llm')
        infer_pkg = types.ModuleType('swift.llm.infer')
        infer_engine_pkg = types.ModuleType('swift.llm.infer.infer_engine')
        infer_engine_mod = types.ModuleType('swift.llm.infer.infer_engine.infer_engine')
        infer_engine_mod.InferEngine = _FakeInferEngine
        vllm_engine_mod = types.ModuleType('swift.llm.infer.infer_engine.vllm_engine')
        vllm_engine_mod.VllmEngine = _FakeVllmEngine
        protocol_mod = types.ModuleType('swift.llm.infer.protocol')
        protocol_mod.RequestConfig = _FakeRequestConfig
        utils_mod = types.ModuleType('swift.utils')
        utils_mod.get_dist_setting = lambda: (0, 0, 1, 1)
        utils_mod.is_dist = lambda: False
        tqdm_mod = types.ModuleType('tqdm')
        tqdm_mod.tqdm = _FakeTqdm
        torch_mod = types.ModuleType('torch')
        torch_mod.inference_mode = lambda: contextlib.nullcontext()

        sys.modules.update({
            'swift': swift_pkg,
            'swift.llm': llm_pkg,
            'swift.llm.infer': infer_pkg,
            'swift.llm.infer.infer_engine': infer_engine_pkg,
            'swift.llm.infer.infer_engine.infer_engine': infer_engine_mod,
            'swift.llm.infer.infer_engine.vllm_engine': vllm_engine_mod,
            'swift.llm.infer.protocol': protocol_mod,
            'swift.utils': utils_mod,
            'tqdm': tqdm_mod,
            'torch': torch_mod,
        })

        patch_module = importlib.import_module('custom_plugins.vllm_infer_pipeline_patch')
        patch_module = importlib.reload(patch_module)
        yield patch_module, vllm_engine_mod.VllmEngine
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


class TestWrapperHelpers(unittest.TestCase):

    def test_parse_wrapper_args(self):
        args, rest = _parse_wrapper_args(
            ['--patched-vllm-encode-num-workers', '8', '--infer_backend', 'vllm'])
        self.assertEqual(args.patched_vllm_encode_num_workers, 8)
        self.assertEqual(rest, ['--infer_backend', 'vllm'])

    def test_has_explicit_option(self):
        self.assertTrue(_has_explicit_option(['--vllm_use_async_engine', 'false'], '--vllm_use_async_engine'))
        self.assertTrue(_has_explicit_option(['--vllm_use_async_engine=true'], '--vllm_use_async_engine'))
        self.assertFalse(_has_explicit_option(['--infer_backend', 'vllm'], '--vllm_use_async_engine'))


class TestPatchModule(unittest.TestCase):

    def test_apply_patch_is_idempotent_and_updates_defaults(self):
        with _stubbed_patch_env() as (patch_module, engine_cls):
            patch_module.apply_patch(encode_num_workers=3, infer_prefetch_size=5)
            first_infer = engine_cls.infer
            patch_module.apply_patch(encode_num_workers=7, infer_prefetch_size=9)
            self.assertIs(first_infer, engine_cls.infer)

            engine = engine_cls(use_async_engine=True)
            self.assertEqual(engine._patched_vllm_encode_num_workers, 7)
            self.assertEqual(engine._patched_vllm_infer_prefetch_size, 9)

    def test_patched_infer_preserves_order_and_non_strict_errors(self):
        with _stubbed_patch_env() as (patch_module, engine_cls):
            patch_module.apply_patch(encode_num_workers=2, infer_prefetch_size=2)
            engine = engine_cls(use_async_engine=True)
            engine.strict = False
            active = {'current': 0, 'max': 0}

            async def fake_infer_async(self, infer_request, request_config=None, *, template=None,
                                       adapter_request=None, pre_infer_hook=None):
                import asyncio

                active['current'] += 1
                active['max'] = max(active['max'], active['current'])
                await asyncio.sleep((3 - infer_request) * 0.01)
                active['current'] -= 1
                if infer_request == 1:
                    raise RuntimeError('boom')
                return f'resp-{infer_request}'

            engine.infer_async = MethodType(fake_infer_async, engine)
            metric = _FakeMetric()
            result = engine.infer(
                [0, 1, 2],
                request_config=_FakeRequestConfig(stream=False),
                metrics=[metric],
                template=_FakeTemplate(),
                use_tqdm=False,
            )

            self.assertEqual(result[0], 'resp-0')
            self.assertIsInstance(result[1], RuntimeError)
            self.assertEqual(result[2], 'resp-2')
            self.assertLessEqual(active['max'], 2)
            self.assertEqual(metric.responses, ['resp-0', 'resp-2'])


if __name__ == '__main__':
    unittest.main()
