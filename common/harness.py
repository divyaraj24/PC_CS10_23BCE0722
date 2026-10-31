"""Benchmark harness: timing, repeats, overhead accounting, shared memory,
peak-memory sampling, CSV writing and environment logging.

Entry scripts must pin BLAS threads BEFORE numpy is imported; `check_env()`
raises if that was not done.
"""
import csv
import os
import platform
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

STUDENT_ID = "23BCE0722"
BATCH_ID = "CS10"
REPEATS = 5
WARMUP = 1

COLUMNS = ["Student_ID", "Batch_ID", "Workload", "Dataset", "Problem_Size",
           "Algorithm", "Parallel_Model", "Hardware", "Resource_Count", "Threads",
           "Execution_Time", "Speedup", "Efficiency", "Communication_Time",
           "Synchronization_Time", "Scheduling_Time", "Memory_Use",
           "Energy_if_available", "Run_ID"]


def check_env():
    bad = [v for v in THREAD_VARS if os.environ.get(v) != "1"]
    if bad:
        raise RuntimeError(f"BLAS thread vars not pinned to 1: {bad}")
    if "numpy" in sys.modules:
        # numpy was imported; env must have been set before. We cannot fully
        # verify ordering here, but entry scripts set it on line 1.
        pass


# --------------------------------------------------------------------------
# Hardware / environment
# --------------------------------------------------------------------------
def _sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NA"


def hardware_string():
    cpu = _sysctl("machdep.cpu.brand_string")
    p, e = _sysctl("hw.perflevel0.physicalcpu"), _sysctl("hw.perflevel1.physicalcpu")
    mem = _sysctl("hw.memsize")
    mem_gb = f"{int(mem) / 2**30:.0f}GB" if mem.isdigit() else "NA"
    return f"{cpu} ({p}P+{e}E cores; {mem_gb} RAM)"


def write_env(path=None):
    import numpy, sklearn, skimage, psutil
    path = Path(path or RESULTS / "env.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    mem = _sysctl("hw.memsize")
    lines = [
        f"CPU model            : {_sysctl('machdep.cpu.brand_string')}",
        f"Logical CPUs         : {_sysctl('hw.ncpu')}",
        f"Performance cores    : {_sysctl('hw.perflevel0.physicalcpu')} ({_sysctl('hw.perflevel0.name')})",
        f"Efficiency cores     : {_sysctl('hw.perflevel1.physicalcpu')} ({_sysctl('hw.perflevel1.name')})",
        f"RAM                  : {int(mem) / 2**30:.1f} GiB" if mem.isdigit() else "RAM: NA",
        f"OS                   : macOS {platform.mac_ver()[0]} (Darwin {platform.release()}, {platform.machine()})",
        f"Python               : {sys.version.split()[0]} ({sys.executable})",
        f"NumPy                : {numpy.__version__}",
        f"scikit-learn         : {sklearn.__version__}",
        f"scikit-image         : {skimage.__version__}",
        f"psutil               : {psutil.__version__}",
        f"multiprocessing start: spawn (macOS default)",
        "BLAS threads         : " + ", ".join(f"{v}={os.environ.get(v)}" for v in THREAD_VARS),
        f"Repeats              : {WARMUP} warm-up (discarded) + {REPEATS} timed per configuration",
        f"Memory sampler       : mode={MEM_MODE}, 20 ms interval (see harness.MemSampler)",
        f"Timer                : time.perf_counter (in-process), time.time (cross-process stamps)",
        f"GPU                  : not used (CPU-only study)",
        f"Energy               : not measured (NA)",
    ]
    try:
        cfg = numpy.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        lines.append(f"NumPy BLAS           : {blas.get('name', 'NA')} {blas.get('version', '')}".rstrip())
    except Exception:
        pass
    pre = path.parent / "preflight.txt"
    if pre.exists():  # pre-flight log for threats-to-validity, written before the sweep
        lines += ["", pre.read_text().rstrip()]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Peak memory (RSS of this process + all children), sampled in a thread
# --------------------------------------------------------------------------
# Sampler modes:
#   "scan"  - psutil children(recursive=True): scans every process on the system
#             (~3 ms per sample on macOS, held under the parent's GIL). Original mode.
#   "track" - RSS of this process + child PIDs registered via track_children()
#             (~0.1 ms per sample). Default.
#   "off"   - no sampling (Memory_Use = parent RSS at start/end only); diagnostics.
MEM_MODE = os.environ.get("CS10_MEM_SAMPLER", "track")
_TRACKED = {}


def track_children(pids):
    """Register worker PIDs (Pool workers / pipeline processes) for memory sampling."""
    import psutil
    for pid in pids:
        try:
            _TRACKED[pid] = psutil.Process(pid)
        except Exception:
            pass


def track_pool(pool):
    track_children([p.pid for p in pool._pool])


class MemSampler:
    def __init__(self, interval=0.02, mode=None):
        import psutil
        self._proc = psutil.Process()
        self.interval = interval
        self.mode = mode or MEM_MODE
        self.peak = 0
        self._stop = threading.Event()

    def _total(self):
        total = 0
        try:
            total = self._proc.memory_info().rss
            if self.mode == "scan":
                kids = self._proc.children(recursive=True)
            elif self.mode == "track":
                kids = list(_TRACKED.values())
            else:
                kids = []
            for c in kids:
                try:
                    total += c.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._total())
            self._stop.wait(self.interval)

    def __enter__(self):
        _TRACKED.clear()
        self.peak = self._total()
        if self.mode != "off":
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.mode != "off":
            self._t.join()
        self.peak = max(self.peak, self._total())
        _TRACKED.clear()

    @property
    def peak_mb(self):
        return self.peak / 2**20


# --------------------------------------------------------------------------
# Overhead accounting
# --------------------------------------------------------------------------
class Overheads:
    """Accumulates Scheduling / Communication / Synchronization seconds."""

    def __init__(self):
        self.sched = 0.0
        self.comm = 0.0
        self.sync = 0.0


def _timed_call(packed):
    fn, arg = packed
    s = time.time()
    r = fn(arg)
    e = time.time()
    return r, os.getpid(), s, e


def timed_map(pool, fn, args, ov):
    """pool.map with per-call overhead decomposition.

    dispatch  = first task start - submit            -> Scheduling
    barrier   = (last end - first start) - max per-worker busy time
                (= time the busiest worker's peers sit idle at the barrier)
                                                     -> Synchronization
    return    = results received - last task end     -> Communication
    """
    t0 = time.time()
    out = pool.map(_timed_call, [(fn, a) for a in args], chunksize=1)
    t1 = time.time()
    starts = [o[2] for o in out]
    ends = [o[3] for o in out]
    busy = {}
    for _, pid, s, e in out:
        busy[pid] = busy.get(pid, 0.0) + (e - s)
    ov.sched += max(min(starts) - t0, 0.0)
    ov.sync += max((max(ends) - min(starts)) - max(busy.values()), 0.0)
    ov.comm += max(t1 - max(ends), 0.0)
    return [o[0] for o in out]


# --------------------------------------------------------------------------
# Shared memory helpers
# --------------------------------------------------------------------------
class SharedArray:
    """numpy array backed by multiprocessing.shared_memory (owner side)."""

    def __init__(self, shape, dtype):
        import numpy as np
        dtype = np.dtype(dtype)
        nbytes = max(int(np.prod(shape)) * dtype.itemsize, 1)
        self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        self.spec = (self.shm.name, tuple(shape), dtype.str)

    @classmethod
    def from_array(cls, a):
        sa = cls(a.shape, a.dtype)
        sa.arr[...] = a
        return sa

    def release(self):
        del self.arr
        self.shm.close()
        self.shm.unlink()


_WORKER_ARRAYS = {}
_WORKER_SHMS = []


def pool_init(specs):
    """Pool initializer: attach named shared arrays once per worker."""
    import numpy as np
    for key, (name, shape, dt) in specs.items():
        shm = shared_memory.SharedMemory(name=name, track=False)
        _WORKER_SHMS.append(shm)
        _WORKER_ARRAYS[key] = np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf)


def warr(key):
    return _WORKER_ARRAYS[key]


# --------------------------------------------------------------------------
# Benchmark driver + CSV
# --------------------------------------------------------------------------
class CsvSink:
    def __init__(self, path, columns):
        self.path = Path(path)
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self):
        if self.path.exists():
            self.path.unlink()

    def write(self, rows):
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)


RAW = CsvSink(RESULTS / "raw.csv", COLUMNS)
CORRECTNESS = CsvSink(RESULTS / "correctness.csv",
                      ["Workload", "Dataset", "Problem_Size", "Algorithm", "Parallel_Model",
                       "Resource_Count", "Run_ID", "Check", "Passed", "Detail"])


class CorrectnessError(AssertionError):
    pass


def check(cond, ctx, run_id, name, detail=""):
    CORRECTNESS.write([{**ctx, "Run_ID": run_id, "Check": name,
                        "Passed": bool(cond), "Detail": detail}])
    if not cond:
        raise CorrectnessError(f"CORRECTNESS FAILURE [{run_id}] {name}: {detail}")


def benchmark(ctx, run_fn, verify_fn=None, repeats=REPEATS, warmup=WARMUP, on_timed=None):
    """Run `run_fn()` warmup+repeats times.

    run_fn() -> dict(exec, comm, sync, sched, result, ...).
    verify_fn(result, run_id) is called on every run (warm-up included).
    on_timed(run_id, out, rep) lets callers write extra per-run CSVs.
    Writes one raw.csv row per timed run. Returns list of outputs (timed only).
    """
    hw = hardware_string()
    base_id = "-".join(str(ctx[k]) for k in ("Workload", "Dataset", "Problem_Size",
                                              "Algorithm", "Parallel_Model")) + f"-w{ctx['Resource_Count']}"
    outs = []
    for rep in range(warmup + repeats):
        timed = rep >= warmup
        run_id = f"{base_id}-r{rep - warmup + 1}" if timed else f"{base_id}-warmup"
        with MemSampler() as mem:
            out = run_fn()
        if verify_fn is not None:
            verify_fn(out["result"], run_id)
        if not timed:
            continue
        RAW.write([{
            "Student_ID": STUDENT_ID, "Batch_ID": BATCH_ID, **ctx, "Hardware": hw,
            "Threads": ctx.get("Threads", 1),
            "Execution_Time": f"{out['exec']:.6f}", "Speedup": "", "Efficiency": "",
            "Communication_Time": f"{out.get('comm', 0.0):.6f}",
            "Synchronization_Time": f"{out.get('sync', 0.0):.6f}",
            "Scheduling_Time": f"{out.get('sched', 0.0):.6f}",
            "Memory_Use": f"{mem.peak_mb:.1f}", "Energy_if_available": "NA",
            "Run_ID": run_id,
        }])
        if on_timed is not None:
            on_timed(run_id, out, rep - warmup + 1)
        outs.append(out)
        print(f"    {run_id:60s} {out['exec']:8.3f}s  mem={mem.peak_mb:7.1f}MB", flush=True)
    return outs


def ctx_keys(ctx):
    return {k: ctx[k] for k in ("Workload", "Dataset", "Problem_Size", "Algorithm",
                                "Parallel_Model", "Resource_Count")}


def split_ranges(n, parts):
    """Split range(n) into `parts` contiguous (start, stop) ranges, near-equal."""
    parts = max(1, min(parts, n))
    q, r = divmod(n, parts)
    out, s = [], 0
    for i in range(parts):
        e = s + q + (1 if i < r else 0)
        out.append((s, e))
        s = e
    return out
