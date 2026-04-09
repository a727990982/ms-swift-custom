#!/usr/bin/env python3
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Run ms-swift GRPO with extra TensorBoard metrics split by a dataset key, '
        'expecting ms-swift to be available via PYTHONPATH and the plugin to be passed via --external_plugins.')
    parser.add_argument(
        '--tb_group_key',
        required=True,
        help='The dataset field name used to split GRPO TensorBoard metrics, e.g. task/domain/source.')
    args, remaining = parser.parse_known_args()

    os.environ['SWIFT_GRPO_TB_GROUP_KEY'] = args.tb_group_key

    from swift.llm.train.rlhf import rlhf_main
    rlhf_main(remaining)


if __name__ == '__main__':
    main()
