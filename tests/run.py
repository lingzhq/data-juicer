# The code is from adapted from
# https://github.com/alibaba/FederatedScope/blob/master/tests/run.py

# Data-Juicer adopts Apache 2.0 license, the original license of this file
# is as follows:
# --------------------------------------------------------
# Copyright (c) Alibaba, Inc. and its affiliates

import argparse
import os
import sys
import unittest

from loguru import logger

from data_juicer.utils.unittest_utils import SKIPPED_TESTS

file_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(file_dir)

parser = argparse.ArgumentParser('test runner')
parser.add_argument('--list_tests', action='store_true', help='list all tests')
parser.add_argument('--pattern', default='test_*.py', help='test file pattern')
parser.add_argument('--test_dir',
                    default='tests',
                    help='directory to be tested')
args = parser.parse_args()


def gather_test_cases(test_dir, pattern, list_tests):
    test_suite = unittest.TestSuite()
    discover = unittest.defaultTestLoader.discover(test_dir,
                                                   pattern=pattern,
                                                   top_level_dir=None)
    print(f'These tests will be skipped due to some reasons: '
          f'{SKIPPED_TESTS.modules}')
    for suite_discovered in discover:

        for test_case in suite_discovered:
            logger.info(f'Prepare for test [{test_case}]')
            # filter out those tests that need to be skipped
            filtered_test_suite = unittest.TestSuite()
            for tc in test_case:
                if type(tc) in SKIPPED_TESTS.modules.values():
                    continue
                filtered_test_suite.addTest(tc)
            if filtered_test_suite.countTestCases() == 0:
                continue

            test_suite.addTest(test_case)
            if hasattr(test_case, '__iter__'):
                for subcase in test_case:
                    if list_tests:
                        print(subcase)
            else:
                if list_tests:
                    print(test_case)
    return test_suite


def main():
    runner = unittest.TextTestRunner()
    test_suite = gather_test_cases(os.path.abspath(args.test_dir),
                                   args.pattern, args.list_tests)
    if not args.list_tests:
        res = runner.run(test_suite)
        if not res.wasSuccessful():
            exit(1)


if __name__ == '__main__':
    main()
