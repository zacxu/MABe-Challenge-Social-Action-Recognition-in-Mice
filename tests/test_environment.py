from __future__ import annotations

from mabe.environment import CORE_DEPENDENCIES, dependency_versions, environment_report


def test_all_core_distributions_have_versions() -> None:
    versions = dependency_versions()
    assert versions.keys() == {dependency.distribution for dependency in CORE_DEPENDENCIES}
    assert all(version for version in versions.values())


def test_environment_report_is_json_compatible_shape() -> None:
    report = environment_report(name="unit-test")
    assert report["name"] == "unit-test"
    assert isinstance(report["python"], str)
    assert isinstance(report["dependencies"], dict)
