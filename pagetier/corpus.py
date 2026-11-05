"""Load a census and decompose it into aligned 2 MB tiles with order pyramids.

A tile is one aligned 2 MB span of one process's address space that holds at
least one resident page. For every architecture order the solver may use, a tile
carries two numbers per aligned sub-span:

  counts[order][tile, i]  how many of its base pages are resident
  legal[order][tile, i]   whether the whole sub-span sits inside a single
                          committed region of uniform protection and backing
                          type, which is what a kernel needs before it may map
                          the span with one entry

Spans with no resident page are never worth mapping (they would cost memory and
save no translation), so only tiles with residency are materialised.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .arch import TILE_ORDER, Arch

PAGE_SHIFT = 12
PAGE_SIZE = 1 << PAGE_SHIFT
TILE_PAGES = 1 << TILE_ORDER
_PROC_STRIDE = 1 << 40


@dataclass
class Corpus:
    """A tiled resident-page census."""

    proc_cls: np.ndarray
    tile_proc: np.ndarray        # int32, which process each tile belongs to
    tile_base: np.ndarray        # int64, first base page of the tile
    resident: np.ndarray         # bool [n_tiles, 512], per-page residency
    shared: np.ndarray           # bool [n_tiles, 512], kernel shared flag
    region_of: np.ndarray        # int32 [n_tiles, 512], covering region or -1
    n_regions: int
    tile_group: np.ndarray       # int64, tiles that share one backing object
    tile_private: np.ndarray     # bool, tile holds private memory
    prot: np.ndarray
    mtype: np.ndarray
    big_proc: np.ndarray         # committed regions of at least 32 MB
    big_base: np.ndarray
    big_npages: np.ndarray

    @property
    def n_tiles(self) -> int:
        return self.resident.shape[0]

    @property
    def resident_pages(self) -> int:
        return int(self.resident.sum())

    def tile_resident(self) -> np.ndarray:
        return self.resident.sum(axis=1).astype(np.int32)

    def pyramid(self, archi: Arch) -> dict:
        """counts and legality for every in-tile order the architecture folds."""
        out = {}
        for order in archi.within_tile():
            if order == 0:
                continue
            width = 1 << order
            groups = TILE_PAGES >> order
            counts = self.resident.reshape(self.n_tiles, groups, width).sum(axis=2)
            gid = self.region_of.reshape(self.n_tiles, groups, width)
            first = gid[:, :, :1]
            legal = (gid >= 0).all(axis=2) & (gid == first).all(axis=2)
            out[order] = (counts.astype(np.int32), legal)
        return out

    def subset(self, tile_mask: np.ndarray) -> "Corpus":
        """A corpus restricted to some tiles, with sharing groups renumbered."""
        _, groups = np.unique(self.tile_group[tile_mask], return_inverse=True)
        return Corpus(
            proc_cls=self.proc_cls,
            tile_proc=self.tile_proc[tile_mask],
            tile_base=self.tile_base[tile_mask],
            resident=self.resident[tile_mask],
            shared=self.shared[tile_mask],
            region_of=self.region_of[tile_mask],
            n_regions=self.n_regions,
            tile_group=groups.astype(np.int64),
            tile_private=self.tile_private[tile_mask],
            prot=self.prot,
            mtype=self.mtype,
            big_proc=self.big_proc,
            big_base=self.big_base,
            big_npages=self.big_npages,
        )

    @property
    def n_groups(self) -> int:
        return int(self.tile_group.max()) + 1 if self.tile_group.size else 0

    def union_bits(self) -> np.ndarray:
        """Per sharing group, which base pages are resident in any member."""
        out = np.zeros((self.n_groups, TILE_PAGES), dtype=bool)
        np.logical_or.at(out, self.tile_group, self.resident)
        return out

    def grouped_pyramid(self, archi: Arch) -> dict:
        """Per sharing group and order, the four quantities the solver needs.

        sum_res    translations the group costs today, summed over every member
        union_res  base pages already resident somewhere, so already paid for
        active     members holding at least one resident page in the span, which
                   is how many translations remain if the span becomes one folio
        legal      the span is legal in every member, the conservative condition
                   under which one folio can serve all of them

        A single-member group reduces to the per-process case exactly: sum_res
        equals union_res and active is one.
        """
        union = self.union_bits()
        groups_of = self.tile_group
        n_groups = self.n_groups
        out = {}
        for order in archi.within_tile():
            if order == 0:
                continue
            width = 1 << order
            cols = TILE_PAGES >> order
            counts_t = self.resident.reshape(self.n_tiles, cols, width).sum(axis=2)
            gid = self.region_of.reshape(self.n_tiles, cols, width)
            legal_t = (gid >= 0).all(axis=2) & (gid == gid[:, :, :1]).all(axis=2)

            sum_res = np.zeros((n_groups, cols), dtype=np.int64)
            np.add.at(sum_res, groups_of, counts_t.astype(np.int64))
            active = np.zeros((n_groups, cols), dtype=np.int64)
            np.add.at(active, groups_of, (counts_t > 0).astype(np.int64))
            legal = np.ones((n_groups, cols), dtype=bool)
            np.logical_and.at(legal, groups_of, legal_t)
            union_res = union.reshape(n_groups, cols, width).sum(axis=2).astype(np.int64)
            out[order] = {"sum_res": sum_res, "union_res": union_res,
                          "active": active, "legal": legal}
        return out

    def sharing_report(self) -> dict:
        """Describe the inferred sharing groups and check them against the OS.

        The groups are inferred from the address and the backing type, never
        from the kernel's shared flag, so that flag is an independent check: a
        page the solver treats as shared should be one the system also calls
        shared.
        """
        sizes = np.bincount(self.tile_group)
        members = sizes[self.tile_group]
        multi = members > 1
        res = self.resident
        res_multi = int(res[multi].sum())
        res_single = int(res[~multi].sum())
        agree = int((res[multi] & self.shared[multi]).sum())
        single_shared = int((res[~multi] & self.shared[~multi]).sum())
        return {
            "groups": self.n_groups,
            "multi_member_groups": int((sizes > 1).sum()),
            "tiles_in_multi": int(multi.sum()),
            "largest_group": int(sizes.max()),
            "resident_in_multi": res_multi,
            "resident_in_single": res_single,
            "multi_called_shared_by_os": agree,
            "single_called_shared_by_os": single_shared,
            "private_tiles": int(self.tile_private.sum()),
        }

    def above_tile_table(self, archi: Arch) -> dict:
        """Spans larger than one tile, with exact legality.

        A span is legal only when a single committed region of uniform
        protection and backing type contains all of it. The census records every
        committed region at least 32 MB long whether or not it holds a resident
        page, so this is decided rather than bounded. Spans with no resident
        page are not enumerated: mapping one would cost memory and save no
        translation.

        Returns {order: {"inv", "legal", "resident", "span_pages"}} where inv
        maps each tile row to its span index.
        """
        tile_index = self.tile_base >> TILE_ORDER
        out = {}
        for order in archi.above_tile():
            shift = order - TILE_ORDER
            span_pages = 1 << order
            span_key = self.tile_proc.astype(np.int64) * _PROC_STRIDE + (tile_index >> shift)
            uniq, inv = np.unique(span_key, return_inverse=True)
            span_proc = (uniq // _PROC_STRIDE).astype(np.int32)
            span_base = (uniq % _PROC_STRIDE) * span_pages
            legal = np.zeros(uniq.size, dtype=bool)
            for s in range(uniq.size):
                sel = self.big_proc == span_proc[s]
                if not sel.any():
                    continue
                base = self.big_base[sel]
                end = base + self.big_npages[sel]
                legal[s] = bool(np.any((base <= span_base[s]) & (end >= span_base[s] + span_pages)))
            resident = np.zeros(uniq.size, dtype=np.int64)
            np.add.at(resident, inv, self.tile_resident().astype(np.int64))
            out[order] = {"inv": inv.astype(np.int64), "legal": legal,
                          "resident": resident, "span_pages": span_pages}
        return out

    def legal_above_tile(self, archi: Arch) -> dict:
        """{order: (spans_with_residency, legal_spans)} from above_tile_table."""
        return {o: (int(t["legal"].size), int(t["legal"].sum()))
                for o, t in self.above_tile_table(archi).items()}


def load(path: str) -> Corpus:
    with np.load(path, allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    return build(fields)


def build(f: dict) -> Corpus:
    proc_cls = f["proc_cls"]
    proc_of = f["proc_of"]
    npages = f["npages"]
    mtype = f["mtype"]
    prot = f["prot"]
    n_regions = proc_of.size

    # Rebuild absolute region bases from the per-process deltas.
    base_page = np.empty(n_regions, dtype=np.int64)
    starts = np.searchsorted(proc_of, np.arange(proc_cls.size + 1))
    for p in range(proc_cls.size):
        lo, hi = starts[p], starts[p + 1]
        base_page[lo:hi] = np.cumsum(f["base_delta"][lo:hi])

    # Resident pages, with the region that holds each and the kernel's shared bit.
    res_region = f["res_region"]
    bit_off = f["bit_off"]
    pages, owners, shared = [], [], []
    for i in range(res_region.size):
        ri = int(res_region[i])
        lo, hi = int(bit_off[i]), int(bit_off[i + 1])
        n = int(npages[ri])
        res = np.unpackbits(f["res_bits"][lo:hi])[:n].astype(bool)
        shr = np.unpackbits(f["shared_bits"][lo:hi])[:n].astype(bool)
        hit = np.flatnonzero(res)
        if hit.size == 0:
            continue
        pages.append(base_page[ri] + hit.astype(np.int64))
        owners.append(np.full(hit.size, ri, dtype=np.int32))
        shared.append(shr[hit])
    all_pages = np.concatenate(pages)
    all_owner = np.concatenate(owners)
    all_shared = np.concatenate(shared)
    all_proc = proc_of[all_owner].astype(np.int64)

    tile_key = all_proc * _PROC_STRIDE + (all_pages >> TILE_ORDER)
    uniq, row = np.unique(tile_key, return_inverse=True)
    n_tiles = uniq.size
    tile_proc = (uniq // _PROC_STRIDE).astype(np.int32)
    tile_base = ((uniq % _PROC_STRIDE) << TILE_ORDER).astype(np.int64)

    within = (all_pages & (TILE_PAGES - 1)).astype(np.int64)
    resident = np.zeros((n_tiles, TILE_PAGES), dtype=bool)
    resident[row, within] = True
    shared_map = np.zeros((n_tiles, TILE_PAGES), dtype=bool)
    shared_map[row, within] = all_shared

    # Attribute every page of every tile to its committed region. Regions are
    # visited per process through a sorted-base lookup, so the cost follows the
    # number of tiles rather than the size of the committed address space.
    region_of = np.full((n_tiles, TILE_PAGES), -1, dtype=np.int32)
    order_by_proc = np.lexsort((base_page, proc_of))
    sorted_proc = proc_of[order_by_proc]
    sorted_base = base_page[order_by_proc]
    sorted_end = sorted_base + npages[order_by_proc]
    proc_starts = np.searchsorted(sorted_proc, np.arange(proc_cls.size + 1))

    for t in range(n_tiles):
        p = tile_proc[t]
        lo_i, hi_i = proc_starts[p], proc_starts[p + 1]
        tb = tile_base[t]
        te = tb + TILE_PAGES
        seg_base = sorted_base[lo_i:hi_i]
        seg_end = sorted_end[lo_i:hi_i]
        first = max(int(np.searchsorted(seg_base, tb, side="right")) - 1, 0)
        last = int(np.searchsorted(seg_base, te, side="left"))
        for j in range(first, last):
            a = max(tb, seg_base[j])
            b = min(te, seg_end[j])
            if b > a:
                region_of[t, a - tb : b - tb] = order_by_proc[lo_i + j]

    # A tile backed only by image or section mappings can be shared with another
    # process at the same address: in one boot session Windows keeps an image at
    # one base in every process that loads it. Private memory never is.
    attributed = region_of >= 0
    kinds = np.where(attributed, mtype[np.clip(region_of, 0, None)], -1)
    private = (kinds == 2).any(axis=1)
    group_key = tile_base * 1024 + np.where(private, tile_proc + 1, 0)
    _, tile_group = np.unique(group_key, return_inverse=True)

    return Corpus(
        proc_cls=proc_cls,
        tile_proc=tile_proc,
        tile_base=tile_base,
        resident=resident,
        shared=shared_map,
        region_of=region_of,
        n_regions=n_regions,
        tile_group=tile_group.astype(np.int64),
        tile_private=private,
        prot=prot,
        mtype=mtype,
        big_proc=f["big_proc"],
        big_base=f["big_base"],
        big_npages=f["big_npages"],
    )


def default_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "census.npz")
