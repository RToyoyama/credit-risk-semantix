import sys

import credit_risk


def test_python_version_is_311():
    assert sys.version_info[:2] == (3, 11)


def test_package_exposes_version():
    assert credit_risk.__version__ == "1.0.0"
