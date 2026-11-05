"""Exact page-order selection under a memory budget.

The decision for one aligned span is not independent of its neighbours: mapping
a 2 MB span with one entry subsumes every 64 KB span inside it. The candidate
spans of an address space therefore form a laminar family (a tree by
containment), and a solution is a cut through that tree. On a tree the problem
has an exact linear-time solution for a fixed price on memory:

    f(node) = min( 1 + lam * (pages(node) - resident(node))   if the node is
                                                              legal and holds a
                                                              resident page,
                   sum of f(child) )

with a leaf worth one entry when resident and nothing when not. Sweeping the
price with the Eisner and Severance parametric method enumerates every vertex of
the lower convex envelope of (extra memory, translations), and each vertex is an
exactly optimal integral solution at its own memory cost. Between two vertices
the envelope is a certified lower bound, and a bounded knapsack over the spans
that differ between the bracketing vertices closes the remaining gap.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from .arch import TILE_ORDER, Arch
from .corpus import Corpus

PAGE_SIZE = 4096
TILE_PAGES = 1 << TILE_ORDER


@dataclass
class Solution:
    """One cut, recorded per independent unit so solutions can be recombined."""

    entries: np.ndarray     # int64 per unit
    waste: np.ndarray       # int64 per unit, extra base pages made resident
    lam: float

    @property
    def total_entries(self) -> int:
        return int(self.entries.sum())

    @property
    def total_waste(self) -> int:
        return int(self.waste.sum())

    def key(self) -> tuple:
        return (self.total_waste, self.total_entries)


class Model:
    """The laminar candidate tree for one architecture over one census.

    Spans are keyed by sharing group rather than by process. Two processes that
    map the same image at the same address share the physical pages, so the
    memory a folio costs is paid once for the group while the translations it
    saves accrue to every member. That asymmetry is what a per-process scan
    cannot see, and it is also what stops a density threshold from being the
    right order: the cost depends on the union of the members' resident pages
    and the benefit on their sum.
    """

    def __init__(self, corpus: Corpus, archi: Arch, verbose: bool = False):
        self.corpus = corpus
        self.arch = archi
        self.in_tile = [o for o in archi.within_tile() if o > 0]
        self.grouped = corpus.grouped_pyramid(archi)
        self.above = corpus.legal_above_tile(archi)
        self.resident_pages = corpus.resident_pages
        self.n_units = corpus.n_groups
        self.levels = self._build_levels()
        if verbose:
            for lv in self.levels:
                print("   order %2d: %d group spans with residency, %d legal"
                      % (lv["order"], lv["nz"].size, int(lv["ok"].sum())), flush=True)

    def _build_levels(self) -> list:
        """Compress each order to the group spans that hold a resident page.

        A span with no resident page in any member is worth no translation and
        can never be worth mapping, so it contributes nothing to its parent.
        """
        ng = self.corpus.n_groups
        levels = []
        for order in self.in_tile:
            d = self.grouped[order]
            cols = TILE_PAGES >> order
            flat = d["sum_res"].ravel()
            nz = np.flatnonzero(flat > 0)
            active = d["active"].ravel()[nz].astype(np.float64)
            levels.append({
                "order": order,
                "cols": cols,
                "nz": nz,
                "sum_res": flat[nz].astype(np.float64),
                "active": active,
                "absent": (float(1 << order) - d["union_res"].ravel()[nz]).astype(np.float64),
                "ok": d["legal"].ravel()[nz] & (active > 0),
            })
        for i in range(len(levels) - 1):
            lo, hi = levels[i], levels[i + 1]
            fan = 1 << (hi["order"] - lo["order"])
            group = lo["nz"] // lo["cols"]
            within = lo["nz"] % lo["cols"]
            parent_flat = group * hi["cols"] + within // fan
            pos = np.full(ng * hi["cols"], -1, dtype=np.int64)
            pos[hi["nz"]] = np.arange(hi["nz"].size)
            lo["parent"] = pos[parent_flat]
            if np.any(lo["parent"] < 0):
                raise AssertionError("a group span with residency has no parent span")
        if levels:
            top = levels[-1]
            top["unit"] = (top["nz"] // top["cols"]).astype(np.int64)
        return levels

    def solve(self, lam: float, prefer_collapse: bool = False) -> Solution:
        """Minimise translations + lam * extra pages exactly, bottom up."""
        ent = wst = None
        for i, lv in enumerate(self.levels):
            size = lv["nz"].size
            if i == 0:
                ent_nc = lv["sum_res"]
                wst_nc = np.zeros(size)
            else:
                prev = self.levels[i - 1]
                ent_nc = np.bincount(prev["parent"], weights=ent, minlength=size)
                wst_nc = np.bincount(prev["parent"], weights=wst, minlength=size)
            score_nc = ent_nc + lam * wst_nc
            score_c = lv["active"] + lam * lv["absent"]
            take = lv["ok"] & ((score_c <= score_nc) if prefer_collapse else (score_c < score_nc))
            ent = np.where(take, lv["active"], ent_nc)
            wst = np.where(take, lv["absent"], wst_nc)

        n = self.n_units
        if not self.levels:
            unit_ent = np.zeros(n)
            np.add.at(unit_ent, self.corpus.tile_group, self.corpus.tile_resident())
            unit_wst = np.zeros(n)
        else:
            top = self.levels[-1]
            unit_ent = np.bincount(top["unit"], weights=ent, minlength=n)
            unit_wst = np.bincount(top["unit"], weights=wst, minlength=n)
        return Solution(entries=np.rint(unit_ent).astype(np.int64),
                        waste=np.rint(unit_wst).astype(np.int64), lam=lam)

    def above_tile_ceiling(self) -> dict:
        """Bound what the orders above one tile could add, so leaving them out
        of the search is a proof rather than an omission.

        A contiguous-PMD mapping replaces 16 table entries with one, so its
        gain is bounded by the best case below it: every child span already
        mapped with one entry per member. The cost is bounded below by the span
        size minus an upper bound on the pages already resident anywhere in it.
        If the resulting ratio is under the marginal price of the frontier at a
        budget, the price-relaxed optimum at that budget cannot contain one of
        these spans, so the frontier is unchanged.
        """
        out = {}
        table = self.corpus.above_tile_table(self.arch)
        order9 = self.grouped.get(TILE_ORDER)
        for order, t in table.items():
            legal = t["legal"]
            if not legal.any() or order9 is None:
                out[order] = {"legal": int(legal.sum()), "max_ratio": 0.0,
                              "best_case_gain": 0, "min_cost_pages": 0}
                continue
            inv = t["inv"]
            groups = self.corpus.tile_group
            active = order9["active"][:, 0]
            union = order9["union_res"][:, 0]
            gain = np.zeros(legal.size, dtype=np.int64)
            np.add.at(gain, inv, active[groups].astype(np.int64))
            cost = np.zeros(legal.size, dtype=np.int64)
            key = inv.astype(np.int64) * (groups.max() + 1) + groups
            _, first = np.unique(key, return_index=True)
            np.add.at(cost, inv[first], union[groups[first]].astype(np.int64))
            sel = legal & (gain > 0)
            g = np.maximum(gain[sel] - 1, 0)
            cpages = np.maximum(t["span_pages"] - cost[sel], 1)
            ratio = g / cpages
            out[order] = {"legal": int(sel.sum()),
                          "max_ratio": float(ratio.max()) if ratio.size else 0.0,
                          "best_case_gain": int(g.sum()),
                          "min_cost_pages": int(cpages.sum())}
        return out


@dataclass
class Frontier:
    """Vertices of the lower convex envelope of (extra pages, translations)."""

    waste: np.ndarray
    entries: np.ndarray
    solves: int
    solutions: list
    complete: bool = False

    def bound(self, budget_pages: int) -> float:
        """Certified lower bound on translations at an arbitrary budget.

        Every price solved gives a valid Lagrangian bound: a solution priced at
        lam cannot beat g(lam) - lam * budget for any budget. Taking the best
        over the prices actually solved stays valid even when the vertex set is
        incomplete, which chord interpolation does not.
        """
        best = 0.0
        for s in self.solutions:
            lam = s.lam
            if lam <= 0.0:
                cand = float(s.total_entries) if s.total_waste <= budget_pages else 0.0
            else:
                cand = s.total_entries + lam * (s.total_waste - budget_pages)
            best = max(best, cand)
        return best

    def achievable(self, budget_pages: int) -> tuple:
        """The best vertex that fits the budget: (waste, entries, index)."""
        ok = np.flatnonzero(self.waste <= budget_pages)
        j = int(ok[-1]) if ok.size else 0
        return int(self.waste[j]), int(self.entries[j]), j


def frontier(model: Model, max_solves: int = 900, tol: float = 1e-9,
             verbose: bool = False) -> Frontier:
    """Enumerate the envelope vertices by parametric search on the price.

    Bisecting the price is exponential; tying two adjacent candidate solutions
    gives the single price at which they are indifferent, and solving there
    either produces a new vertex or proves the two are neighbours. Segments are
    refined widest-first so a solve budget is spent where the envelope is still
    coarse, rather than depth-first down one branch.
    """
    # At zero price a tie means a collapse that costs memory and saves no
    # translation, so the strict rule is the right one at both ends: it gives the
    # fewest translations for the least memory that reaches them.
    lo = model.solve(0.0, prefer_collapse=False)              # cheapest in entries
    hi = model.solve(float(1 << 40), prefer_collapse=False)   # cheapest in memory
    found = {lo.key(): lo, hi.key(): hi}
    solves = 2
    queue = []
    seq = itertools.count()

    def push(a, b):
        dw = a.total_waste - b.total_waste
        de = b.total_entries - a.total_entries
        if dw > 0 and de > 0:
            heapq.heappush(queue, (-(dw * de), next(seq), a, b))

    push(lo, hi)
    while queue and solves < max_solves:
        _, _, a, b = heapq.heappop(queue)
        dw = a.total_waste - b.total_waste
        de = b.total_entries - a.total_entries
        if dw <= 0 or de <= 0:
            continue
        lam = de / dw
        mid = model.solve(lam, prefer_collapse=False)
        solves += 1
        f_mid = mid.total_entries + lam * mid.total_waste
        f_a = a.total_entries + lam * a.total_waste
        if f_mid < f_a - tol * max(1.0, abs(f_a)):
            found[mid.key()] = mid
            push(a, mid)
            push(mid, b)
            if verbose and len(found) % 50 == 0:
                print("   %d vertices, %d solves" % (len(found), solves), flush=True)
    # Drop any point another point beats on both axes. A tie-break inside a solve
    # can produce one, and a dominated point is never an envelope vertex.
    keys = []
    floor = None
    for key in sorted(found):
        if floor is None or key[1] < floor:
            keys.append(key)
            floor = key[1]
    waste = np.array([k[0] for k in keys], dtype=np.int64)
    entries = np.array([k[1] for k in keys], dtype=np.int64)
    return Frontier(waste=waste, entries=entries, solves=solves,
                    solutions=[found[k] for k in keys], complete=not queue)


def refine_at_budget(front: Frontier, budget_pages: int,
                     max_cells: int = 40_000_000) -> dict:
    """Close the integrality gap between the two bracketing vertices.

    As the price falls, every independent unit moves weakly towards more memory
    and fewer translations, so the units that differ between two adjacent
    vertices each offer exactly one upgrade with a non-negative cost and a
    non-negative gain. Choosing among them is a bounded knapsack, solved exactly
    when it fits in the cell budget.
    """
    w0, e0, j = front.achievable(budget_pages)
    out = {"vertex_waste": w0, "vertex_entries": e0,
           "bound": front.bound(budget_pages), "refined": False, "exact": True,
           "entries": e0, "waste": w0, "items": 0, "cells": 0}
    if j + 1 >= len(front.solutions):
        return out
    a = front.solutions[j]
    b = front.solutions[j + 1]
    cost = b.waste - a.waste
    gain = a.entries - b.entries
    sel = (cost != 0) | (gain != 0)
    cost = cost[sel]
    gain = gain[sel]
    free = int(gain[(cost <= 0) & (gain > 0)].sum())
    keep = (cost > 0) & (gain > 0)
    cost, gain = cost[keep], gain[keep]
    residual = budget_pages - w0
    out["items"] = int(cost.size)
    if free:
        out.update(entries=e0 - free, refined=True)
        e0 = e0 - free
    if residual <= 0 or cost.size == 0:
        return out
    cells = int(cost.size) * (int(residual) + 1)
    out["cells"] = cells
    if cells <= max_cells:
        dp = np.zeros(int(residual) + 1, dtype=np.int64)
        for c, g in zip(cost.tolist(), gain.tolist()):
            if c > residual:
                continue
            np.maximum(dp[c:], dp[:-c] + g, out=dp[c:])
        best = int(dp.max())
        used = int(np.flatnonzero(dp == best)[0])
        out.update(refined=True, exact=True, entries=e0 - best, waste=w0 + used)
        return out
    # Too wide for an exact table. Ratio order is optimal for the relaxed
    # problem, so greedy lands within one upgrade of the envelope chord, and the
    # chord is reported alongside as the certificate.
    order = np.argsort(-(gain / cost))
    room = int(residual)
    best = 0
    used = 0
    for i in order.tolist():
        c = int(cost[i])
        if c <= room:
            room -= c
            used += c
            best += int(gain[i])
    out.update(refined=True, exact=False, entries=e0 - best, waste=w0 + used)
    return out


def summarise(front: Frontier, budgets_bytes: list) -> list:
    rows = []
    for budget in budgets_bytes:
        pages = budget // PAGE_SIZE
        r = refine_at_budget(front, pages)
        rows.append({
            "budget_bytes": int(budget),
            "entries": int(r["entries"]),
            "vertex_entries": int(r["vertex_entries"]),
            "vertex_waste_bytes": int(r["vertex_waste"]) * PAGE_SIZE,
            "bound": float(r["bound"]),
            "refined": bool(r["refined"]),
            "exact": bool(r["exact"]),
            "waste_bytes": int(r["waste"]) * PAGE_SIZE,
            "refine_items": int(r["items"]),
        })
    return rows
