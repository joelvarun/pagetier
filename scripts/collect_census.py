"""Collect a resident-page census from this machine into data/census.npz."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pagetier import census, corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=corpus.default_path())
    ap.add_argument("--max-processes", type=int, default=None)
    args = ap.parse_args()
    try:
        payload = census.collect(max_processes=args.max_processes)
    except census.CensusUnavailable as exc:
        print("census unavailable: %s" % exc)
        return 2
    census.save(payload, args.out)
    size = os.path.getsize(args.out if args.out.endswith(".npz") else args.out + ".npz")
    print("wrote %s (%.1f KB)" % (args.out, size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
