#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def _inject_external_plugin(argv, plugin_path):
    plugin_path = str(plugin_path)
    if '--external_plugins' not in argv:
        return [*argv, '--external_plugins', plugin_path]

    idx = argv.index('--external_plugins')
    end = idx + 1
    while end < len(argv) and not argv[end].startswith('--'):
        end += 1
    current_values = argv[idx + 1:end]
    if plugin_path in current_values:
        return argv
    return [*argv[:end], plugin_path, *argv[end:]]


def main():
    parser = argparse.ArgumentParser(
        description='Run ms-swift GRPO with extra TensorBoard metrics split by a dataset key.')
    parser.add_argument(
        '--tb_group_key',
        required=True,
        help='The dataset field name used to split GRPO TensorBoard metrics, e.g. task/domain/source.')
    args, remaining = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / 'ms-swift-3.12.0'
    plugin_path = repo_root / 'custom_plugins' / 'grpo_tensorboard_by_key.py'

    if not plugin_path.exists():
        raise FileNotFoundError(f'Plugin file not found: {plugin_path}')

    os.environ['SWIFT_GRPO_TB_GROUP_KEY'] = args.tb_group_key
    sys.path.insert(0, str(package_root))

    patched_argv = _inject_external_plugin(remaining, plugin_path)

    from swift.llm.train.rlhf import rlhf_main
    rlhf_main(patched_argv)


if __name__ == '__main__':
    main()
