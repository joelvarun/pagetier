"""Solver checks that do not depend on the solver being right."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pagetier import arch, corpus, exact, policy, verify

CENSUS = corpus.default_path()
pytestmark = pytest.mark.skipif(not os.path.exists(CENSUS),
                                reason="data/census.npz is not present")


@pytest.fixture(scope="module")
def census():
    return corpus.load(CENSUS)


def test_recurrence_matches_exhaustive_enumeration():
    """Every laminar cut enumerated, against one bottom-up pass."""
    result = verify.check_recurrence(trials=120)
    assert result["cuts_enumerated"] > 1000
    assert result["max_abs_error"] == 0.0


@pytest.mark.parametrize("key", ["x86_64", "arm64_4k"])
def test_vectorised_solver_matches_scalar_reference(census, key):
    model = exact.Model(census, arch.ARCHS[key])
    result = verify.check_vectorised(model, sample=60)
    assert result["max_entry_error"] == 0
    assert result["max_waste_error"] == 0


@pytest.mark.parametrize("key", ["x86_64", "arm64_4k"])
def test_frontier_matches_exact_budgeted_table(census, key):
    result = verify.check_frontier(census, arch.ARCHS[key], groups=20, budget_pages=1500)
    assert result["complete"]
    assert result["exact_table"] == result["frontier_refined"]


@pytest.mark.parametrize("key", ["x86_64", "arm64_4k"])
def test_frontier_is_convex_and_monotone(census, key):
    model = exact.Model(census, arch.ARCHS[key])
    front = exact.frontier(model, max_solves=40000)
    assert front.complete
    waste, entries = front.waste, front.entries
    assert np.all(np.diff(waste) > 0)
    assert np.all(np.diff(entries) < 0)
    slopes = -np.diff(entries) / np.diff(waste)
    assert np.all(np.diff(slopes) <= 1e-9)


@pytest.mark.parametrize("key", ["x86_64", "arm64_4k"])
def test_bound_is_below_every_feasible_policy(census, key):
    """The Lagrangian bound must not exceed anything actually achievable."""
    a = arch.ARCHS[key]
    model = exact.Model(census, a)
    front = exact.frontier(model, max_solves=40000)
    ev = policy.PolicyEvaluator(census, a)
    for rule in (ev.LOCAL, ev.UNION):
        for pts in (ev.envelope(rule=rule, steps=8),):
            for waste, entries, _ in pts:
                assert entries >= front.bound(waste) - 1e-6


def test_no_huge_pages_matches_the_census(census):
    ev = policy.PolicyEvaluator(census, arch.X86_64)
    assert ev.no_huge_pages().entries == census.resident_pages
    assert ev.no_huge_pages().waste_pages == 0


def test_zero_price_equals_the_shipped_default_on_x86(census):
    """Collapsing everything legal is exactly what max_ptes_none=511 asks for."""
    model = exact.Model(census, arch.X86_64)
    cheapest = model.solve(0.0, prefer_collapse=True)
    default = policy.PolicyEvaluator(census, arch.X86_64).khugepaged(511)
    assert cheapest.total_entries == default.entries
    assert cheapest.total_waste * 4096 == default.waste_bytes


def test_one_gigabyte_tier_has_no_legal_span(census):
    for key in ("x86_64", "arm64_4k"):
        above = census.legal_above_tile(arch.ARCHS[key])
        assert above[18][1] == 0


def test_sharing_groups_agree_with_the_kernel_shared_flag(census):
    """The grouping never reads the shared flag, so it is an independent check."""
    report = census.sharing_report()
    multi = report["multi_called_shared_by_os"] / report["resident_in_multi"]
    single = report["single_called_shared_by_os"] / report["resident_in_single"]
    assert multi > 0.9
    assert single < 0.3
