# conftest.py is based on
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# and
# https://docs.pytest.org/en/latest/deprecations.html#pytest-namespace

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-redundant", action="store_true", default=False, help="Run redandant tests."
    )


def pytest_configure():
    pytest.run_redundant = False


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-redundant"):
        pytest.run_redundant = True
