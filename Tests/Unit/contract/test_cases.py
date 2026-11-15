"""C01–C12 合同案例的软件回归（无真实设备时用 NullHandler/FakeHandler 夹具）。"""

from __future__ import annotations

import pytest

from aieyes.contract import cases

from contract_helpers import run


@pytest.mark.parametrize("case_id", list(cases.CASE_ORDER))
def test_contract_case_passes(case_id: str) -> None:
    outcome = run(cases.run_case(case_id))
    assert outcome.status == "passed", outcome.to_dict()
    assert outcome.checks, outcome.to_dict()


def test_run_all_reports_every_case() -> None:
    outcomes = run(cases.run_all())
    assert [outcome.id for outcome in outcomes] == list(cases.CASE_ORDER)
    assert len(outcomes) == 12
    failed = [outcome.to_dict() for outcome in outcomes if outcome.status == "failed"]
    assert not failed, failed


def test_case_titles_cover_c01_to_c12() -> None:
    assert set(cases.CASE_TITLES) == {f"C{i:02d}" for i in range(1, 13)}
    assert set(cases.CASE_FUNCS) == set(cases.CASE_TITLES)
