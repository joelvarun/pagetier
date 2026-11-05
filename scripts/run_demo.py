"""Run the whole study on the committed census and write docs/results.json."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pagetier import arch, corpus, exact, policy, verify

BUDGETS_MB = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
HEADLINE_MB = 256
PAGE = 4096


def gb(pages):
    return pages * PAGE / 2**30


def marginal_price(front, budget_pages):
    """Slope of the envelope at a budget, in translations saved per extra page."""
    w, e = front.waste, front.entries
    j = int(np.searchsorted(w, budget_pages, side="right"))
    if j >= len(w):
        return 0.0
    if j == 0:
        j = 1
    dw = int(w[j] - w[j - 1])
    de = int(e[j - 1] - e[j])
    return float(de / dw) if dw > 0 else float("inf")


def jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(repr(obj))


def report_census(c):
    share = c.sharing_report()
    distinct = int(c.union_bits().sum())
    print("census: %d processes, %d tiles, %d translations over %d distinct pages "
          "(%.2f GB mapped, %.2f GB physical)"
          % (c.proc_cls.size, c.n_tiles, c.resident_pages, distinct,
             gb(c.resident_pages), gb(distinct)), flush=True)
    print("sharing: %d of %d groups have more than one mapper, largest %d; the system itself "
          "calls %.1f%% of their resident pages shared against %.1f%% elsewhere"
          % (share["multi_member_groups"], share["groups"], share["largest_group"],
             100 * share["multi_called_shared_by_os"] / share["resident_in_multi"],
             100 * share["single_called_shared_by_os"] / share["resident_in_single"]), flush=True)
    return {
        "processes": int(c.proc_cls.size),
        "committed_regions": int(c.n_regions),
        "tiles": int(c.n_tiles),
        "resident_pages": int(c.resident_pages),
        "resident_bytes": int(c.resident_pages) * PAGE,
        "distinct_pages": distinct,
        "distinct_bytes": distinct * PAGE,
        "sharing": share,
    }


def report_legality(c, a):
    legality = {}
    pyr = c.pyramid(a)
    for order in a.within_tile():
        if order == 0:
            continue
        counts, legal = pyr[order]
        live = counts > 0
        legality[order] = {
            "spans_with_residency": int(live.sum()),
            "legal": int((legal & live).sum()),
            "resident_pages_in_legal": int(counts[legal].sum()),
        }
        d = legality[order]
        print("  order %-2d (%-13s) %7d spans with residency, %7d legal (%.1f%%), "
              "covering %7d resident pages (%.1f%%)"
              % (order, arch.order_label(a, order), d["spans_with_residency"], d["legal"],
                 100 * d["legal"] / d["spans_with_residency"], d["resident_pages_in_legal"],
                 100 * d["resident_pages_in_legal"] / c.resident_pages), flush=True)
    above = c.legal_above_tile(a)
    for order, (spans, legal) in sorted(above.items()):
        note = "  (tier unreachable, left out of the search)" if legal == 0 else ""
        print("  order %-2d (%-13s) %7d spans with residency, %7d legal%s"
              % (order, arch.order_label(a, order), spans, legal, note), flush=True)
    return legality, above


def run_arch(c, key, steps, do_verify):
    a = arch.ARCHS[key]
    print("\n=== %s, orders that fold into one translation: %s"
          % (a.label, ", ".join(a.order_labels)), flush=True)
    ev = policy.PolicyEvaluator(c, a)
    model = exact.Model(c, a)
    legality, above = report_legality(c, a)

    nothing = ev.no_huge_pages()
    default = ev.khugepaged(policy.KHUGEPAGED_MAX_PTES_NONE_DEFAULT)
    print("  no huge pages                          %9d translations, 0 extra"
          % nothing.entries, flush=True)
    print("  khugepaged default (max_ptes_none=511) %9d translations, %.2f GB extra"
          % (default.entries, default.waste_bytes / 2**30), flush=True)

    t = time.time()
    env_local = ev.envelope(rule=ev.LOCAL, steps=steps)
    env_union = ev.envelope(rule=ev.UNION, steps=steps)
    print("  threshold envelopes: %d per-mm points, %d union points (%.1fs)"
          % (len(env_local), len(env_union), time.time() - t), flush=True)

    t = time.time()
    front = exact.frontier(model, max_solves=40000)
    print("  parametric frontier: %d vertices from %d solves, complete=%s (%.1fs)"
          % (len(front.waste), front.solves, front.complete, time.time() - t), flush=True)

    ceiling = model.above_tile_ceiling()
    for order, d in sorted(ceiling.items()):
        if d["legal"]:
            print("  order %-2d (%-13s) left out of the search: %d legal spans, best possible "
                  "%.4f translations per extra page"
                  % (order, arch.order_label(a, order), d["legal"], d["max_ratio"]), flush=True)

    floor_waste = int(front.waste[-1]) * PAGE
    dead = None
    if int(front.entries[-1]) == default.entries:
        dead = default.waste_bytes - floor_waste
        print("  the fewest translations this ladder can reach is %d, and pagetier reaches it "
              "for %.2f GB, so %.2f GB of the default's request buys no translation at all"
              % (int(front.entries[-1]), floor_waste / 2**30, dead / 2**30), flush=True)

    rows = []
    for mb in BUDGETS_MB:
        pages = mb * 2**20 // PAGE
        wl, el, liml = policy.envelope_at(env_local, pages)
        wu, eu, limu = policy.envelope_at(env_union, pages)
        r = exact.refine_at_budget(front, pages)
        rows.append({
            "budget_mb": mb,
            "per_mm_entries": int(el), "per_mm_bytes": int(wl) * PAGE,
            "per_mm_limits": {str(k): v for k, v in liml.items()},
            "union_entries": int(eu), "union_bytes": int(wu) * PAGE,
            "union_limits": {str(k): v for k, v in limu.items()},
            "exact_entries": int(r["entries"]), "exact_bytes": int(r["waste"]) * PAGE,
            "bound": float(r["bound"]), "exact_refine": bool(r["exact"]),
            "marginal_price": marginal_price(front, pages),
        })
        print("  %5d MB  per-mm %9d | union %9d | pagetier %9d  (bound %9.0f, gap %.4f%%)"
              "  marginal price %.2f translations per page"
              % (mb, el, eu, r["entries"], r["bound"],
                 100 * (r["entries"] - r["bound"]) / max(1.0, r["bound"]),
                 rows[-1]["marginal_price"]), flush=True)

    block = {
        "label": a.label,
        "order_labels": list(a.order_labels),
        "orders": list(a.orders),
        "legality": {str(k): v for k, v in legality.items()},
        "above_tile": {str(k): list(v) for k, v in above.items()},
        "above_tile_ceiling": {str(k): v for k, v in ceiling.items()},
        "no_huge_pages": int(nothing.entries),
        "khugepaged_default": {"entries": int(default.entries),
                               "bytes": int(default.waste_bytes),
                               "dead_bytes": int(dead) if dead is not None else None},
        "floor": {"entries": int(front.entries[-1]), "bytes": floor_waste},
        "frontier": {"waste_bytes": (front.waste * PAGE).tolist(),
                     "entries": front.entries.tolist(),
                     "vertices": int(len(front.waste)),
                     "solves": int(front.solves),
                     "complete": bool(front.complete)},
        "envelope_local": [[int(w) * PAGE, int(e)] for w, e, _ in env_local],
        "envelope_union": [[int(w) * PAGE, int(e)] for w, e, _ in env_union],
        "budget_rows": rows,
    }
    checks = None
    if do_verify:
        vec = verify.check_vectorised(model, sample=150)
        fr = verify.check_frontier(c, a, groups=30, budget_pages=2500)
        print("  solver check: scalar reference on %d groups, max error %d; exact budgeted "
              "table on a %d group slice gives %d against %d from the frontier"
              % (vec["groups_checked"], vec["max_entry_error"], fr["groups"],
                 fr["exact_table"], fr["frontier_refined"]), flush=True)
        checks = {"vectorised": vec, "frontier": fr}
    return block, checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", default=corpus.default_path())
    ap.add_argument("--out", default=os.path.join("docs", "results.json"))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("loading census from %s" % args.census, flush=True)
    c = corpus.load(args.census)
    out = {"census": report_census(c), "budgets_mb": BUDGETS_MB,
           "headline_mb": HEADLINE_MB, "arch": {}}

    do_verify = not args.skip_verify
    if do_verify:
        print("\nverifying the solver", flush=True)
        rec = verify.check_recurrence(trials=150)
        print("  tree recurrence against %d enumerated laminar cuts: max error %g"
              % (rec["cuts_enumerated"], rec["max_abs_error"]), flush=True)
        out["verify"] = {"recurrence": rec, "per_arch": {}}

    for key in ("x86_64", "arm64_4k"):
        block, checks = run_arch(c, key, args.steps, do_verify)
        out["arch"][key] = block
        if checks is not None:
            out["verify"]["per_arch"][key] = checks

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=jsonable)
    print("\nwrote %s in %.1fs total" % (args.out, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
