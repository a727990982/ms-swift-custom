#!/usr/bin/env python3
import argparse
import os
import sys


def _add_repo_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    ms_swift_root = os.path.join(repo_root, 'ms-swift-3.12.0')
    for path in [repo_root, ms_swift_root]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root, ms_swift_root


def _has_explicit_option(argv, option_name):
    return any(arg == option_name or arg.startswith(f'{option_name}=') for arg in argv)


def _parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--patched-vllm-encode-num-workers', type=int, default=None)
    parser.add_argument('--patched-vllm-infer-prefetch-size', type=int, default=None)
    return parser.parse_known_args(argv)


def main():
    wrapper_args, forwarded_argv = _parse_wrapper_args(sys.argv[1:])
    _add_repo_paths()

    from swift.cli.main import prepare_config_args
    from swift.llm import InferArguments, infer_main
    from swift.utils import get_logger, parse_args

    from custom_plugins.vllm_infer_pipeline_patch import apply_patch

    logger = get_logger()
    prepare_config_args(forwarded_argv)
    explicit_async = _has_explicit_option(forwarded_argv, '--vllm_use_async_engine')
    args, remaining_argv = parse_args(InferArguments, forwarded_argv)
    if remaining_argv:
        if getattr(args, 'ignore_args_error', False):
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')

    if args.infer_backend == 'vllm':
        if not explicit_async and not args.eval_human:
            args.vllm_use_async_engine = True
        apply_patch(
            encode_num_workers=wrapper_args.patched_vllm_encode_num_workers,
            infer_prefetch_size=wrapper_args.patched_vllm_infer_prefetch_size,
        )

    infer_main(args)


if __name__ == '__main__':
    main()
