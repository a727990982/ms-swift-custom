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
    parser.add_argument('--multi_teacher_model', '--multi-teacher-model', nargs='+', default=None)
    parser.add_argument('--multi_teacher_model_type', '--multi-teacher-model-type', nargs='*', default=None)
    parser.add_argument('--multi_teacher_model_revision', '--multi-teacher-model-revision', nargs='*', default=None)
    parser.add_argument('--multi_teacher_weights', '--multi-teacher-weights', nargs='*', type=float, default=None)
    parser.add_argument('--teacher_seq_kd_index', '--teacher-seq-kd-index', type=int, default=0)
    return parser.parse_known_args(argv)


def main():
    wrapper_args, forwarded_argv = _parse_wrapper_args(sys.argv[1:])
    _add_repo_paths()

    from swift.cli.main import prepare_config_args
    from swift.llm.train.rlhf import rlhf_main

    prepare_config_args(forwarded_argv)

    if wrapper_args.multi_teacher_model:
        from custom_plugins.multi_teacher_gkd_patch import apply_patch

        if not _has_explicit_option(forwarded_argv, '--teacher_model'):
            forwarded_argv = ['--teacher_model', wrapper_args.multi_teacher_model[0], *forwarded_argv]

        apply_patch(
            teacher_models=wrapper_args.multi_teacher_model,
            teacher_model_types=wrapper_args.multi_teacher_model_type,
            teacher_model_revisions=wrapper_args.multi_teacher_model_revision,
            teacher_weights=wrapper_args.multi_teacher_weights,
            teacher_seq_kd_index=wrapper_args.teacher_seq_kd_index,
        )

    rlhf_main(forwarded_argv)


if __name__ == '__main__':
    main()
