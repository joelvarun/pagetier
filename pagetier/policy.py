"""The policies a kernel actually ships, evaluated on the same census.

Linux decides huge-page promotion with a small set of global knobs:

  transparent_hugepage/enabled                       on or off, per boot
  transparent_hugepage/hugepages-<size>kB/enabled    one switch per folio order
  khugepaged/max_ptes_none                           how empty a span may be and
                                                     still be collapsed, default
                                                     511 of 512, so one resident
                                                     base page is enough

Every one of those is a single number applied to every span in the system. The
family modelled here is a strict superset of what the knobs can express: an
independent density threshold per order, with the largest enabled order tried
first, which is the order khugepaged and the fault handler both prefer. Beating
the lower envelope of that whole family is therefore a stronger statement than
beating any one shipped configuration.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .arch import TILE_ORDER, Arch
from .corpus import Corpus

TILE_PAGES = 1 << TILE_ORDER
PAGE_SIZE = 4096

# khugepaged's shipped default: collapse when at most 511 of the 512 base pages
# in the span are absent, that is, when at least one is resident.
KHUGEPAGED_MAX_PTES_NONE_DEFAULT = 511


@dataclass
class PolicyResult:
    label: str
    entries: int
    waste_pages: int
    detail: dict

    @property
    def waste_bytes(self) -> int:
        return self.waste_pages * PAGE_SIZE


class PolicyEvaluator:
    """Evaluate threshold policies on the same sharing-group model.

    Two decision rules are offered. The first is the one that ships: khugepaged
    walks one mm at a time, so the span it judges is one process's mapping and
    the number it compares is that process's own count of absent page table
    entries. The second is a stronger rule that no kernel implements: judge the
    span on the union of every member's residency, which is what the memory
    actually costs. Reporting both separates the part of the result that comes
    from seeing the sharing from the part that comes from solving exactly.
    """

    LOCAL = "per-mm"
    UNION = "union"

    def __init__(self, corpus: Corpus, archi: Arch):
        self.corpus = corpus
        self.arch = archi
        self.local = corpus.pyramid(archi)
        self.grouped = corpus.grouped_pyramid(archi)
        self.above = corpus.above_tile_table(archi)
        self.resident_pages = corpus.resident_pages
        self.orders = [o for o in archi.within_tile() if o > 0]

    def no_huge_pages(self) -> PolicyResult:
        return PolicyResult("no huge pages", self.resident_pages, 0, {})

    def evaluate(self, limits: dict, rule: str = LOCAL, label: str = "") -> PolicyResult:
        """limits maps an order to the largest absent fraction it tolerates."""
        ng = self.corpus.n_groups
        groups_of = self.corpus.tile_group
        entries = 0
        waste = 0
        covered = 0
        busy = np.zeros((ng, 1), dtype=bool)
        prev_order = TILE_ORDER

        for order in sorted(self.orders, reverse=True):
            cols = TILE_PAGES >> order
            if order < prev_order:
                busy = np.repeat(busy, 1 << (prev_order - order), axis=1)
            prev_order = order
            if order not in limits:
                continue
            d = self.grouped[order]
            pages = 1 << order
            allow = limits[order] * pages
            if rule == self.UNION:
                want = (pages - d["union_res"]) <= allow
            else:
                counts_t, _ = self.local[order]
                hit = (counts_t > 0) & ((pages - counts_t) <= allow)
                want = np.zeros((ng, cols), dtype=bool)
                np.logical_or.at(want, groups_of, hit)
            take = d["legal"] & (d["active"] > 0) & want & ~busy
            if not take.any():
                continue
            entries += int(d["active"][take].sum())
            waste += int((pages - d["union_res"])[take].sum())
            covered += int(d["sum_res"][take].sum())
            busy = busy | take

        entries += self.resident_pages - covered
        return PolicyResult(label or "%s %s" % (rule, limits), entries, waste, dict(limits))

    def khugepaged(self, max_ptes_none: int = KHUGEPAGED_MAX_PTES_NONE_DEFAULT) -> PolicyResult:
        """The shipped 2 MB-only policy, parameterised by max_ptes_none."""
        return self.evaluate({TILE_ORDER: max_ptes_none / TILE_PAGES}, rule=self.LOCAL,
                             label="khugepaged max_ptes_none=%d" % max_ptes_none)

    def usable_orders(self) -> list:
        """Orders with at least one legal span. The rest cannot be used by any
        policy, so dropping them from the family is a proof, not a shortcut."""
        return [o for o in self.orders
                if bool((self.grouped[o]["legal"] & (self.grouped[o]["active"] > 0)).any())]

    def knob_family(self, steps: int = 32) -> list:
        orders = self.usable_orders()
        out = []
        for size in range(1, len(orders) + 1):
            fine = {1: steps * 4, 2: steps}.get(size, max(8, steps // 3))
            grid = sorted({i / fine for i in range(fine + 1)})
            for subset in itertools.combinations(orders, size):
                for combo in itertools.product(grid, repeat=size):
                    out.append(dict(zip(subset, combo)))
        return out

    def envelope(self, rule: str = LOCAL, steps: int = 32, verbose: bool = False) -> list:
        """Lower-left envelope of the whole family: list of (waste, entries, limits)."""
        points = [(0, self.resident_pages, {})]
        configs = self.knob_family(steps=steps)
        for i, limits in enumerate(configs):
            r = self.evaluate(limits, rule=rule)
            points.append((r.waste_pages, r.entries, limits))
            if verbose and (i + 1) % 400 == 0:
                print("   %d of %d settings" % (i + 1, len(configs)), flush=True)
        points.sort(key=lambda p: (p[0], p[1]))
        best = []
        floor = None
        for waste, entries, limits in points:
            if floor is None or entries < floor:
                best.append((waste, entries, limits))
                floor = entries
        return best


def envelope_at(envelope: list, budget_pages: int) -> tuple:
    """Best setting in an envelope that fits a budget: (waste, entries, limits)."""
    ok = [p for p in envelope if p[0] <= budget_pages]
    return ok[-1] if ok else envelope[0]
