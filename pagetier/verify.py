"""Checks that the fast solver computes what the slow definition says.

Three independent checks:

  check_recurrence      on synthetic groups small enough to enumerate every
                        laminar cut, the tree recurrence must equal the
                        exhaustive minimum
  check_vectorised      on the real census, a scalar Python solve of a sampled
                        group must equal the vectorised solver's output for it
  check_frontier        on a small slice of the real census, an exact budgeted
                        table built from each group's full Pareto set must equal
                        the parametric frontier plus its refinement
"""

from __future__ import annotations

import random

import numpy as np

from .arch import TILE_ORDER, Arch
from .corpus import Corpus

TILE_PAGES = 1 << TILE_ORDER


def _synthetic_group(width: int, members: int, rng: random.Random) -> dict:
    """Build one group from explicit member bitmaps and a region layout.

    Legality is derived from a per-page region id, so it is monotone the way the
    real thing is: a sub-span of a uniform span is uniform.
    """
    bitmaps = [[rng.random() < 0.45 for _ in range(width)] for _ in range(members)]
    region = []
    current = rng.randrange(3)
    for _ in range(width):
        if rng.random() < 0.25:
            current = rng.randrange(3)
        region.append(-1 if rng.random() < 0.1 else current)
    return {"bitmaps": bitmaps, "region": region, "width": width}


def _node_stats(group: dict, order: int, index: int) -> tuple:
    lo = index << order
    hi = lo + (1 << order)
    counts = [sum(bm[lo:hi]) for bm in group["bitmaps"]]
    sum_res = sum(counts)
    union = sum(1 for p in range(lo, hi) if any(bm[p] for bm in group["bitmaps"]))
    active = sum(1 for c in counts if c > 0)
    span = group["region"][lo:hi]
    legal = span[0] >= 0 and all(r == span[0] for r in span)
    return sum_res, union, active, legal


def _recurrence(group: dict, orders: list, order_idx: int, index: int, lam: float) -> tuple:
    order = orders[order_idx]
    sum_res, union, active, legal = _node_stats(group, order, index)
    if order_idx == 0:
        ent_nc, wst_nc = sum_res, 0
    else:
        fan = 1 << (order - orders[order_idx - 1])
        ent_nc = wst_nc = 0
        for k in range(fan):
            e, w = _recurrence(group, orders, order_idx - 1, index * fan + k, lam)
            ent_nc += e
            wst_nc += w
    if legal and active > 0:
        absent = (1 << order) - union
        if active + lam * absent < ent_nc + lam * wst_nc:
            return active, absent
    return ent_nc, wst_nc


def _all_cuts(group: dict, orders: list, order_idx: int, index: int) -> list:
    order = orders[order_idx]
    sum_res, union, active, legal = _node_stats(group, order, index)
    out = []
    if legal and active > 0:
        out.append((active, (1 << order) - union))
    if order_idx == 0:
        out.append((sum_res, 0))
        return out
    fan = 1 << (order - orders[order_idx - 1])
    combos = [(0, 0)]
    for k in range(fan):
        child = _all_cuts(group, orders, order_idx - 1, index * fan + k)
        combos = [(e1 + e2, w1 + w2) for e1, w1 in combos for e2, w2 in child]
    out.extend(combos)
    return out


def check_recurrence(trials: int = 200, seed: int = 11) -> dict:
    """Tree recurrence against exhaustive enumeration of every laminar cut."""
    rng = random.Random(seed)
    worst = 0.0
    cuts = 0
    for _ in range(trials):
        orders = rng.choice([[0, 1, 2], [0, 2, 3], [1, 3], [0, 1, 2, 3]])
        width = 1 << orders[-1]
        group = _synthetic_group(width, rng.randrange(1, 5), rng)
        options = _all_cuts(group, orders, len(orders) - 1, 0)
        cuts += len(options)
        for lam in (0.0, 0.01, 0.1, 0.5, 1.0, 3.0, 50.0):
            ent, wst = _recurrence(group, orders, len(orders) - 1, 0, lam)
            got = ent + lam * wst
            best = min(e + lam * w for e, w in options)
            worst = max(worst, abs(got - best))
    return {"trials": trials, "cuts_enumerated": cuts, "max_abs_error": worst,
            "ok": worst == 0.0}


def check_vectorised(model, lams=(0.0, 0.02, 0.2, 2.0, 40.0), sample: int = 400,
                     seed: int = 7) -> dict:
    """Scalar solve of sampled groups against the vectorised solver."""
    rng = np.random.default_rng(seed)
    grouped = model.grouped
    orders = [o for o in model.arch.within_tile() if o > 0]
    units = rng.choice(model.n_units, size=min(sample, model.n_units), replace=False)
    worst_e = worst_w = 0
    for lam in lams:
        sol = model.solve(lam)
        for g in units.tolist():
            ent = wst = None
            for i, order in enumerate(orders):
                cols = TILE_PAGES >> order
                d = grouped[order]
                if i == 0:
                    ent_nc = [int(d["sum_res"][g, j]) for j in range(cols)]
                    wst_nc = [0] * cols
                else:
                    fan = 1 << (order - orders[i - 1])
                    ent_nc = [int(sum(ent[j * fan:(j + 1) * fan])) for j in range(cols)]
                    wst_nc = [int(sum(wst[j * fan:(j + 1) * fan])) for j in range(cols)]
                ent, wst = [], []
                for j in range(cols):
                    active = int(d["active"][g, j])
                    absent = (1 << order) - int(d["union_res"][g, j])
                    ok = bool(d["legal"][g, j]) and active > 0
                    if ok and active + lam * absent < ent_nc[j] + lam * wst_nc[j]:
                        ent.append(active)
                        wst.append(absent)
                    else:
                        ent.append(ent_nc[j])
                        wst.append(wst_nc[j])
            worst_e = max(worst_e, abs(ent[0] - int(sol.entries[g])))
            worst_w = max(worst_w, abs(wst[0] - int(sol.waste[g])))
    return {"groups_checked": int(units.size), "prices": list(lams),
            "max_entry_error": int(worst_e), "max_waste_error": int(worst_w),
            "ok": worst_e == 0 and worst_w == 0}


def group_pareto(model, g: int) -> np.ndarray:
    """Exact minimum translations for every extra-page count, for one group.

    Built with a budget-indexed tree table rather than a price, so it shares no
    code path with the parametric solver it is used to check.
    """
    orders = [o for o in model.arch.within_tile() if o > 0]
    grouped = model.grouped
    INF = np.iinfo(np.int64).max // 4
    table = None
    for i, order in enumerate(orders):
        cols = TILE_PAGES >> order
        pages = 1 << order
        d = grouped[order]
        new = np.full((cols, pages + 1), INF, dtype=np.int64)
        for j in range(cols):
            if i == 0:
                new[j, 0] = int(d["sum_res"][g, j])
            else:
                fan = 1 << (order - orders[i - 1])
                acc = np.zeros(1, dtype=np.int64)
                for k in range(fan):
                    child = table[j * fan + k]
                    conv = np.full(acc.size + child.size - 1, INF, dtype=np.int64)
                    for w, v in enumerate(acc.tolist()):
                        if v >= INF:
                            continue
                        cand = np.where(child >= INF, INF, child + v)
                        np.minimum(conv[w:w + child.size], cand, out=conv[w:w + child.size])
                    acc = np.minimum.accumulate(conv)[: pages + 1]
                new[j, : acc.size] = acc
            active = int(d["active"][g, j])
            absent = pages - int(d["union_res"][g, j])
            if bool(d["legal"][g, j]) and active > 0:
                new[j, absent] = min(int(new[j, absent]), active)
            new[j] = np.minimum.accumulate(new[j])
        table = new
    return table[0]


def check_frontier(corpus: Corpus, archi: Arch, groups: int = 40,
                   budget_pages: int = 3000, seed: int = 5) -> dict:
    """Exact budgeted table over a slice of the census against the frontier."""
    from . import exact

    rng = np.random.default_rng(seed)
    sizes = np.bincount(corpus.tile_group, minlength=corpus.n_groups)
    pick = rng.choice(np.flatnonzero(sizes > 0), size=groups, replace=False)
    sub = corpus.subset(np.isin(corpus.tile_group, pick))
    model = exact.Model(sub, archi)

    INF = np.iinfo(np.int64).max // 4
    dp = np.zeros(budget_pages + 1, dtype=np.int64)
    for g in range(model.n_units):
        curve = group_pareto(model, g)[: budget_pages + 1]
        nxt = np.full(budget_pages + 1, INF, dtype=np.int64)
        for w, v in enumerate(curve.tolist()):
            if v >= INF:
                continue
            tail = dp[: budget_pages + 1 - w]
            cand = np.where(tail >= INF, INF, tail + v)
            np.minimum(nxt[w:], cand, out=nxt[w:])
        dp = np.minimum.accumulate(nxt)
    truth = int(dp[budget_pages])

    front = exact.frontier(model, max_solves=20000)
    got = exact.refine_at_budget(front, budget_pages)
    return {"groups": int(groups), "budget_pages": int(budget_pages),
            "exact_table": truth, "frontier_refined": int(got["entries"]),
            "bound": float(got["bound"]), "complete": bool(front.complete),
            "ok": truth == int(got["entries"])}
