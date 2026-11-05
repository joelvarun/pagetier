"""Which aligned page orders a given architecture can turn into one translation.

An order k means an aligned span of 2**k base pages (4 KB each). Only the orders
listed here collapse into a single TLB entry on that architecture:

  * x86-64 has no contiguous-PTE hint, so only the real page sizes count:
    4 KB (order 0), 2 MB (order 9) and 1 GB (order 18).
  * ARM64 with a 4 KB granule adds the contiguous-PTE and contiguous-PMD hints,
    which let the MMU fold 16 adjacent entries into one: 64 KB (order 4) and
    32 MB (order 13).

Linux allocates folios at many more orders than these (multi-size THP covers
every order from 1 to 9), but an order the hardware cannot fold still needs one
TLB entry per base page, so it does not reduce the metric here.
"""

from __future__ import annotations

from dataclasses import dataclass

TILE_ORDER = 9  # the census is decomposed into aligned 2 MB tiles


@dataclass(frozen=True)
class Arch:
    key: str
    label: str
    orders: tuple           # orders that fold into one translation, ascending
    order_labels: tuple

    def within_tile(self) -> tuple:
        return tuple(o for o in self.orders if o <= TILE_ORDER)

    def above_tile(self) -> tuple:
        return tuple(o for o in self.orders if o > TILE_ORDER)


X86_64 = Arch(
    key="x86_64",
    label="x86-64",
    orders=(0, 9, 18),
    order_labels=("4KB", "2MB", "1GB"),
)

ARM64_4K = Arch(
    key="arm64_4k",
    label="ARM64 (4KB granule)",
    orders=(0, 4, 9, 13, 18),
    order_labels=("4KB", "64KB contpte", "2MB", "32MB contpmd", "1GB"),
)

ARCHS = {a.key: a for a in (X86_64, ARM64_4K)}


def order_label(archi: Arch, order: int) -> str:
    return archi.order_labels[archi.orders.index(order)]
