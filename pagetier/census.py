"""Harvest a resident-page census from the running Windows system.

For every process this account can open, the census records

  * the committed virtual-memory regions that hold at least one resident page,
    with their protection and backing type, and
  * a packed bitmap of which pages inside each region are resident.

That is exactly the input a huge-page collapse policy sees: residency plus the
region boundaries that decide whether an aligned span may become one folio.
Nothing is read out of another process's memory; only the page maps are queried.
"""

from __future__ import annotations

import bisect
import ctypes
import ctypes.wintypes as wt
import os
from dataclasses import dataclass, field

import numpy as np

PAGE_SHIFT = 12
PAGE_SIZE = 1 << PAGE_SHIFT

MEM_COMMIT = 0x1000
MEM_IMAGE = 0x1000000
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
ERROR_BAD_LENGTH = 24

TYPE_CODES = {MEM_IMAGE: 0, MEM_MAPPED: 1, MEM_PRIVATE: 2}
BIG_REGION_PAGES = 1 << 13  # 32 MB, the smallest above-tile order of interest
TYPE_NAMES = ["image", "mapped", "private"]

# Protection values are normalised to a small dense code so the committed
# corpus does not carry raw Win32 constants.
PROT_ORDER = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x40000000]
PROT_NAMES = {
    0x01: "noaccess", 0x02: "readonly", 0x04: "readwrite", 0x08: "writecopy",
    0x10: "execute", 0x20: "execute_read", 0x40: "execute_readwrite",
    0x80: "execute_writecopy", 0x100: "guard", 0x200: "nocache",
}

# Only widely distributed binary names are kept in the committed corpus. Any
# other image is recorded as "other" so no local path or private tool leaks.
PUBLIC_IMAGE_NAMES = {
    "chrome.exe", "msedge.exe", "firefox.exe", "explorer.exe", "svchost.exe",
    "code.exe", "python.exe", "pythonw.exe", "node.exe", "powershell.exe",
    "pwsh.exe", "cmd.exe", "conhost.exe", "dwm.exe", "csrss.exe", "winlogon.exe",
    "services.exe", "lsass.exe", "spoolsv.exe", "taskhostw.exe", "sihost.exe",
    "runtimebroker.exe", "searchindexer.exe", "searchhost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "textinputhost.exe", "applicationframehost.exe",
    "audiodg.exe", "fontdrvhost.exe", "wininit.exe", "smss.exe", "registry",
    "memory compression", "system", "msmpeng.exe", "nissrv.exe", "securityhealthservice.exe",
    "git.exe", "bash.exe", "sh.exe", "wsl.exe", "wslhost.exe", "docker.exe",
    "msedgewebview2.exe", "widgets.exe", "lockapp.exe", "systemsettings.exe",
    "openconsole.exe", "windowsterminal.exe", "notepad.exe", "mspaint.exe",
}

_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ULONG_PTR = ctypes.c_size_t


class _MemoryBasicInformation(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("_pad0", wt.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
        ("_pad1", wt.DWORD),
    ]


_k32.VirtualQueryEx.argtypes = [
    wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(_MemoryBasicInformation), ctypes.c_size_t
]
_k32.VirtualQueryEx.restype = ctypes.c_size_t
_k32.OpenProcess.restype = wt.HANDLE


class CensusUnavailable(RuntimeError):
    """Raised when the host cannot supply a resident-page census."""


@dataclass
class ProcessCensus:
    cls: str
    regions: list = field(default_factory=list)   # (base_page, npages, prot_code, type_code)
    res_index: list = field(default_factory=list)  # which regions carry a bitmap
    bitmaps: list = field(default_factory=list)    # packed residency, one per res_index
    shared: list = field(default_factory=list)     # packed shared flag, same shape


def enum_pids() -> list:
    cap = 4096
    while True:
        buf = (wt.DWORD * cap)()
        need = wt.DWORD()
        if not _psapi.EnumProcesses(ctypes.byref(buf), ctypes.sizeof(buf), ctypes.byref(need)):
            raise ctypes.WinError(ctypes.get_last_error())
        got = need.value // ctypes.sizeof(wt.DWORD)
        if got < cap:
            return list(buf[:got])
        cap *= 2


def _image_basename(handle) -> str:
    buf = ctypes.create_unicode_buffer(1024)
    size = wt.DWORD(1024)
    if _k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
        return os.path.basename(buf.value).lower()
    return ""


def query_working_set(handle):
    """Resident virtual page numbers of a process, with the kernel's shared flag.

    A PSAPI working-set block packs the protection in bits 0 to 4, the share
    count in bits 5 to 7, the shared flag in bit 8 and the virtual page number
    above bit 11. The shared flag is the system's own answer to whether the
    backing page is shared with another process, which is used later to check
    the inferred sharing groups rather than to build them.
    """
    cap = 1 << 16
    for _ in range(16):
        nbytes = ctypes.sizeof(_ULONG_PTR) * (cap + 1)
        buf = ctypes.create_string_buffer(nbytes)
        if _psapi.QueryWorkingSet(handle, buf, nbytes):
            raw = np.frombuffer(buf.raw, dtype=np.uint64, count=cap + 1)
            count = int(raw[0])
            blocks = raw[1 : 1 + count]
            return (blocks >> PAGE_SHIFT).copy(), ((blocks >> 8) & 1).astype(bool)
        if ctypes.get_last_error() != ERROR_BAD_LENGTH:
            return None, None
        raw = np.frombuffer(buf.raw, dtype=np.uint64, count=1)
        cap = max(cap * 2, int(raw[0]) + 4096)
    return None, None


def query_regions(handle) -> list:
    """Return every mapped region as (base_page, npages, state, protect, type)."""
    out = []
    addr = 0
    mbi = _MemoryBasicInformation()
    limit = 1 << 47
    while addr < limit:
        if _k32.VirtualQueryEx(handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)) == 0:
            break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize
        if size == 0:
            break
        out.append((base >> PAGE_SHIFT, size >> PAGE_SHIFT, mbi.State, mbi.Protect, mbi.Type))
        addr = base + size
    return out


def _prot_code(protect: int) -> int:
    base = protect & 0xFF
    extra = protect & 0xFFFFFF00
    try:
        idx = PROT_ORDER.index(base)
    except ValueError:
        idx = len(PROT_ORDER)
    # Fold PAGE_GUARD and PAGE_NOCACHE into distinct codes: they split a VMA too.
    return idx * 4 + (1 if extra & 0x100 else 0) + (2 if extra & 0x200 else 0)


def collect(max_processes: int | None = None, verbose: bool = True) -> dict:
    """Collect the census from the live system."""
    if os.name != "nt":
        raise CensusUnavailable(
            "the census reader uses QueryWorkingSet and VirtualQueryEx, which exist only on Windows; "
            "use the committed corpus in data/ instead"
        )
    pids = enum_pids()
    processes = []
    big_regions = []
    skipped = 0
    for pid in pids:
        handle = _k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            skipped += 1
            continue
        try:
            name = _image_basename(handle)
            resident, shared_flag = query_working_set(handle)
            if resident is None or resident.size == 0:
                skipped += 1
                continue
            regions = query_regions(handle)
        finally:
            _k32.CloseHandle(handle)

        committed = [r for r in regions if r[2] == MEM_COMMIT and r[4] in TYPE_CODES]
        if not committed:
            skipped += 1
            continue
        # VirtualQueryEx already coalesces pages of identical attributes, so a
        # committed region is by construction a uniform span. Keep every region
        # big enough to host a span above tile size, residency or not, so that
        # the 32 MB and 1 GB tiers can be decided exactly rather than bounded.
        for r in committed:
            if r[1] >= BIG_REGION_PAGES:
                big_regions.append((len(processes), int(r[0]), int(r[1]), _prot_code(r[3]), TYPE_CODES[r[4]]))
        committed.sort()
        bases = [r[0] for r in committed]
        resident, uniq_pick = np.unique(resident, return_index=True)
        shared_flag = shared_flag[uniq_pick]

        # Attribute each resident page to its committed region.
        idx = np.searchsorted(np.asarray(bases, dtype=np.uint64), resident, side="right") - 1
        keep = idx >= 0
        idx = idx[keep]
        pages = resident[keep]
        flags = shared_flag[keep]
        ends = np.asarray([r[0] + r[1] for r in committed], dtype=np.uint64)
        inside = pages < ends[idx]
        idx = idx[inside]
        pages = pages[inside]
        flags = flags[inside]
        if pages.size == 0:
            skipped += 1
            continue

        proc = ProcessCensus(cls=name if name in PUBLIC_IMAGE_NAMES else "other")
        order = np.argsort(idx, kind="stable")
        idx = idx[order]
        pages = pages[order]
        flags = flags[order]
        bounds = np.searchsorted(idx, np.arange(len(committed) + 1))
        # Every committed region is recorded, residency or not: a span is legal
        # only when one committed region of uniform attributes contains all of
        # it, so regions without residency still decide the answer.
        for ri, region in enumerate(committed):
            base_page, npages = region[0], region[1]
            proc.regions.append((int(base_page), int(npages), _prot_code(region[3]), TYPE_CODES[region[4]]))
            lo, hi = bounds[ri], bounds[ri + 1]
            if hi <= lo:
                continue
            res = np.zeros(npages, dtype=bool)
            shr = np.zeros(npages, dtype=bool)
            off = (pages[lo:hi] - base_page).astype(np.int64)
            res[off] = True
            shr[off] = flags[lo:hi]
            proc.res_index.append(len(proc.regions) - 1)
            proc.bitmaps.append(np.packbits(res))
            proc.shared.append(np.packbits(shr))
        if proc.bitmaps:
            processes.append(proc)
        if max_processes is not None and len(processes) >= max_processes:
            break

    if not processes:
        raise CensusUnavailable("no process could be opened for a working-set query")
    big_regions = [b for b in big_regions if b[0] < len(processes)]
    if verbose:
        total = sum(int(np.unpackbits(b).sum()) for p in processes for b in p.bitmaps)
        print(
            "census: %d processes, %d committed regions (%d with residency), "
            "%d resident pages (%.2f GB)"
            % (
                len(processes),
                sum(len(p.regions) for p in processes),
                sum(len(p.bitmaps) for p in processes),
                total,
                total * PAGE_SIZE / 2**30,
            ),
            flush=True,
        )
        print("census: %d processes could not be opened or had no working set" % skipped, flush=True)
    return pack(processes, big_regions)


def pack(processes: list, big_regions: list) -> dict:
    """Flatten per-process censuses into the arrays stored on disk.

    Region bases are stored as deltas within a process so the archive stays
    small: an address space is enumerated in increasing address order.
    """
    proc_of, delta, npages, prot, mtype = [], [], [], [], []
    res_region, bit_off = [], []
    res_chunks, shr_chunks = [], []
    offset = 0
    region_base = 0
    for pi, proc in enumerate(processes):
        previous = 0
        first = region_base
        for (b, n, pr, ty) in proc.regions:
            proc_of.append(pi)
            delta.append(b - previous)
            previous = b
            npages.append(n)
            prot.append(pr)
            mtype.append(ty)
        for ri, res, shr in zip(proc.res_index, proc.bitmaps, proc.shared):
            res_region.append(first + ri)
            bit_off.append(offset)
            res_chunks.append(res)
            shr_chunks.append(shr)
            offset += res.size
        region_base += len(proc.regions)
    bit_off.append(offset)
    big = np.asarray(big_regions, dtype=np.int64).reshape(-1, 5)
    return {
        "proc_cls": np.array([p.cls for p in processes]),
        "proc_of": np.asarray(proc_of, dtype=np.int32),
        "base_delta": np.asarray(delta, dtype=np.int64),
        "npages": np.asarray(npages, dtype=np.int64),
        "prot": np.asarray(prot, dtype=np.int16),
        "mtype": np.asarray(mtype, dtype=np.int8),
        "res_region": np.asarray(res_region, dtype=np.int64),
        "bit_off": np.asarray(bit_off, dtype=np.int64),
        "res_bits": np.concatenate(res_chunks) if res_chunks else np.zeros(0, dtype=np.uint8),
        "shared_bits": np.concatenate(shr_chunks) if shr_chunks else np.zeros(0, dtype=np.uint8),
        "big_proc": big[:, 0].astype(np.int32),
        "big_base": big[:, 1].astype(np.int64),
        "big_npages": big[:, 2].astype(np.int64),
        "big_prot": big[:, 3].astype(np.int16),
        "big_mtype": big[:, 4].astype(np.int8),
    }


def save(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)
