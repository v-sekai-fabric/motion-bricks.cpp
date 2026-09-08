"""Measure MuJoCo Warp step cost for the G1 with and without VRM hull collision.

SPDX-License-Identifier: Apache-2.0

This is the number that gates the #174 sweep commitment, so the harness is
built to make a contended or cold-clock measurement fail loudly rather than
quietly bias the table:

* the target GPU is selected **by name**, never by index -- on this box
  ``nvidia-smi`` and PyTorch disagree on device order, and a previous session
  measurement reported the 3090 faster than the 4090 purely from cold clocks;
* GPU utilisation and memory are sampled **before and after** every timed run,
  and a run whose GPU was busy is discarded and retried rather than averaged in;
* warmup steps run before every timed window;
* all conditions are printed alongside the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

TARGET_GPU = os.environ.get("BENCH_TARGET_GPU", "4090")


def gpu_state() -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,utilization.gpu,memory.used,clocks.sm",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    rows = []
    for line in out.splitlines():
        i, name, util, mem, clk = [p.strip() for p in line.split(",")]
        rows.append({"index": int(i), "name": name, "util": int(util),
                     "mem_mib": int(mem), "sm_mhz": int(clk)})
    return rows


def target_row(rows: list[dict]) -> dict:
    for r in rows:
        if TARGET_GPU in r["name"]:
            return r
    raise SystemExit(f"no GPU matching {TARGET_GPU!r}: {rows}")


BASELINE_PIDS: set[int] = set()


def snapshot_pids(gpu_index: int) -> set[int]:
    """PIDs already on the GPU before we start.

    The 4090 on this box is the DISPLAY gpu -- explorer.exe, ShellHost.exe,
    Parsec and dwm are permanently attached to it.  Treating any foreign PID as
    contention rejects every run forever.  So we baseline the desktop set at
    startup and flag only processes that appear afterwards.
    """
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout
    pids = set()
    for line in out.splitlines():
        line = line.strip()
        try:
            pids.add(int(line.split(",")[0]))
        except (ValueError, IndexError):
            pass
    return pids


def foreign_procs(gpu_index: int, my_pid: int) -> list[tuple[int, int]]:
    """Compute processes that appeared on the target GPU AFTER we started.

    Raw ``utilization.gpu`` cannot be used as a contention signal here: once
    the timed loop starts, the load being measured is *our own*.  What matters
    is whether some other job (a Blender render, another fork's training run)
    joined the device mid-measurement.  Returns [(pid, used_mib), ...].
    """
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}",
         "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    procs = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "not supported" in line.lower():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            continue
        # used_memory is "[N/A]" for processes we cannot inspect. Those still
        # contend for the GPU -- record them with mem=-1 rather than dropping
        # them, which is what let a competing job through undetected.
        try:
            mem = int(parts[1])
        except (ValueError, IndexError):
            mem = -1
        if pid != my_pid and pid not in BASELINE_PIDS:
            procs.append((pid, mem))
    return procs


def select_gpu() -> dict:
    """Pin CUDA to the target GPU by NAME, then verify."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    rows = gpu_state()
    tgt = target_row(rows)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(tgt["index"])
    os.environ["_BENCH_GPU_INDEX"] = str(tgt["index"])
    return tgt


def run_case(xml: str, nworld: int, steps: int, warmup: int,
             busy_util: int, mem_ceiling: int = 8000,
             retries: int = 6) -> dict:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)

    gpu_index = int(os.environ.get("_BENCH_GPU_INDEX", "0"))
    my_pid = os.getpid()

    for attempt in range(retries):
        # Warmup first: it triggers Warp JIT compilation, which would otherwise
        # be counted as load and would also pollute the pre-run idle check.
        for _ in range(warmup):
            mjw.step(m, d)
        wp.synchronize()

        before = target_row(gpu_state())
        fbefore = foreign_procs(gpu_index, my_pid)
        # `before` is sampled after OUR warmup has drained, so residual high
        # utilisation or near-full VRAM is somebody else's load.
        busy_now = before["util"] > busy_util or before["mem_mib"] > mem_ceiling
        if (fbefore or busy_now) and attempt < retries - 1:
            time.sleep(15)
            continue

        t0 = time.perf_counter()
        for _ in range(steps):
            mjw.step(m, d)
        wp.synchronize()
        dt = time.perf_counter() - t0

        after = target_row(gpu_state())
        fafter = foreign_procs(gpu_index, my_pid)
        peak = after["mem_mib"]

        # mujoco_warp exposes the contact count as `nacon` (NOT `ncon`, which is
        # the CPU MjData field name and does not exist on the warp Data object).
        # It is a total across all worlds.
        try:
            ncon = int(d.nacon.numpy()[0])
        except Exception:
            ncon = -1

        contended = bool(fbefore or fafter) or busy_now
        if contended and attempt < retries - 1:
            time.sleep(5)
            continue

        return {
            "nworld": nworld, "steps": steps, "warmup": warmup,
            "ngeom": int(mjm.ngeom), "nconmax": int(d.nconmax)
            if hasattr(d, "nconmax") else None,
            "ncon_total": ncon,
            "ncon_per_env": round(ncon / nworld, 2) if ncon >= 0 else None,
            "ms_per_step": round(dt / steps * 1e3, 4),
            "us_per_step_per_env": round(dt / steps / nworld * 1e6, 3),
            "steps_per_s": round(steps / dt, 1),
            "env_steps_per_s": round(steps * nworld / dt, 0),
            "vram_mib": peak,
            "util_before": before["util"], "util_after": after["util"],
            "sm_mhz_before": before["sm_mhz"], "sm_mhz_after": after["sm_mhz"],
            "foreign_procs": fbefore + fafter,
            "contended": contended, "attempts": attempt + 1,
        }
    return {"nworld": nworld, "error": "foreign GPU processes across all retries"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--envs", type=int, nargs="+",
                    default=[1, 64, 256, 1024, 4096])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--busy-util", type=int, default=25)
    ap.add_argument("--mem-ceiling", type=int, default=8000,
                    help="MiB already in use on the card above which we treat "
                         "the run as contended and retry")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    global BASELINE_PIDS
    tgt = select_gpu()
    BASELINE_PIDS = snapshot_pids(tgt["index"])
    print(f"# target GPU: [{tgt['index']}] {tgt['name']} "
          f"(selected by name, CUDA_DEVICE_ORDER=PCI_BUS_ID)")
    print(f"# baseline GPU PIDs (desktop/display, excluded from contention "
          f"check): {sorted(BASELINE_PIDS)}")
    print(f"# idle check: util={tgt['util']}% mem={tgt['mem_mib']}MiB "
          f"sm={tgt['sm_mhz']}MHz")
    print(f"# TF32: not applicable (MuJoCo Warp kernels, not torch matmul)")
    print(f"# warmup={a.warmup} steps timed={a.steps} busy_util_threshold={a.busy_util}%")

    results = []
    for n in a.envs:
        r = run_case(a.xml, n, a.steps, a.warmup, a.busy_util,
                     mem_ceiling=a.mem_ceiling)
        r["label"] = a.label
        results.append(r)
        print(json.dumps(r), flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
