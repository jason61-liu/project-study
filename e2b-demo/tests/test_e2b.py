import os

import pytest
from e2b import Sandbox


def test_e2b_sdk_is_installed():
    assert callable(Sandbox.create)


@pytest.fixture(scope="module")
def sandbox():
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("需要设置 E2B_API_KEY 才能运行真实沙箱测试")

    instance = Sandbox.create("code-interpreter-v1", timeout=60)
    try:
        yield instance
    finally:
        instance.kill()


@pytest.fixture(scope="module")
def opencode_sandbox():
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("需要设置 E2B_API_KEY 才能运行真实沙箱测试")

    instance = Sandbox.create("opencode", timeout=60)
    try:
        yield instance
    finally:
        instance.kill()


@pytest.mark.integration
def test_run_command(sandbox):
    result = sandbox.commands.run("printf 'hello from e2b'")

    assert result.exit_code == 0
    assert result.stdout == "hello from e2b"


@pytest.mark.integration
def test_write_and_read_file(sandbox):
    path = "/tmp/e2b-test.txt"
    sandbox.files.write(path, "sandbox file content")

    assert sandbox.files.read(path) == "sandbox file content"


@pytest.mark.integration
def test_code_interpreter_has_scientific_packages_and_git(sandbox):
    result = sandbox.commands.run(
        """
set -e
git --version
python3 - <<'PY'
import importlib.metadata

for package in (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "scikit-learn",
    "sympy",
):
    print(f"{package}=={importlib.metadata.version(package)}")
PY
"""
    )

    assert result.exit_code == 0, result.stderr
    for dependency in (
        "git version",
        "numpy==",
        "pandas==",
        "scipy==",
        "matplotlib==",
        "scikit-learn==",
        "sympy==",
    ):
        assert dependency in result.stdout


@pytest.mark.integration
def test_opencode_template_tools_and_scientific_packages(opencode_sandbox):
    result = opencode_sandbox.commands.run(
        """
set -e
opencode --version
git --version
python3 --version
node --version
npm --version
python3 - <<'PY'
import importlib.util

for package in ("numpy", "pandas", "scipy", "matplotlib", "sklearn", "sympy"):
    print(f"{package}={bool(importlib.util.find_spec(package))}")
PY
"""
    )

    assert result.exit_code == 0, result.stderr
    assert "git version" in result.stdout
    for package in ("numpy", "pandas", "scipy", "matplotlib", "sklearn", "sympy"):
        assert f"{package}=False" in result.stdout
