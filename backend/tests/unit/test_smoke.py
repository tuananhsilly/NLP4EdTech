import sys

import learning_agent


def test_package_is_importable():
    assert learning_agent.__doc__


def test_runs_on_python_312():
    assert sys.version_info[:2] == (3, 12)
