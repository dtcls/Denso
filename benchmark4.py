"""
Benchmark for the scripted Panda pick-and-place controller.

Task per episode: pick Obj_A, Obj_C, Obj_D one by one and place each in Frame_B,
perpendicular (90 deg) to Frame_B's primary axis, in separate slots.

An episode counts as a SUCCESS only if, after the robot finishes and the scene
has settled, ALL three objects are inside their slots:
    xy error < 2.5 cm, z error < 2 cm, yaw error (to 90 deg) < 15 deg,
    and none is still held by the gripper.

Usage:
    python benchmark2.py                        # 50 random-layout episodes
    python benchmark2.py --episodes 200 --seed 1000
    python benchmark2.py --fixed-layout         # objects in the fixed row
    python benchmark2.py --out results/run1     # writes run1.csv / run1.json

Metrics: success rate (+95% Wilson CI), per-object success, placement accuracy
(xy / yaw), cycle time per object, total time, and a failure breakdown.
"""

import argparse
import csv
import json
import os
import time
from collections import Counter

import numpy as np

import custom_env4 as ce
from custom_env4 import (CONTROL_FREQ, OBJECT_NAMES, PLACE_XY_TOL, PLACE_YAW_TOL,
                         PLACE_Z_TOL, PickPlaceController, make_env)


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def run_episode(env, seed, settle_steps=30):
    raw = env.reset_raw(seed=seed)
    ctrl = PickPlaceController(env)
    ctrl.reset(raw)

    t0 = time.time()
    while not ctrl.done:
        raw, _, sim_done, _ = env.step_raw(ctrl.step(raw))
        if sim_done and not ctrl.done:
            ctrl._announce(ctrl.FAILED)
            ctrl.fail_reason = ctrl.fail_reason or "simulation horizon reached"
            ctrl.fail_state = ctrl.fail_state or "SIM"
    wall = time.time() - t0

    # let everything settle, then re-measure ALL placed objects (this also
    # catches a later pick knocking over an object placed earlier)
    zero = np.zeros(7)
    for _ in range(settle_steps):
        raw, _, _, _ = env.step_raw(zero)

    rows = {}
    for idx, name in sorted(env.occupied.items()):
        xy, z, yaw = env.placement_error(name, env.slots[idx])
        grasped = env._is_grasped(name)
        ok = (xy < PLACE_XY_TOL and z < PLACE_Z_TOL and yaw < PLACE_YAW_TOL
              and not grasped)
        rows[name] = dict(slot=idx, xy_cm=xy * 100, z_cm=z * 100,
                          yaw_deg=float(np.degrees(yaw)), ok=bool(ok))
    for name in OBJECT_NAMES:
        rows.setdefault(name, dict(slot=None, xy_cm=np.nan, z_cm=np.nan,
                                   yaw_deg=np.nan, ok=False))

    success = bool(ctrl.success) and all(r["ok"] for r in rows.values())
    return dict(
        seed=seed, success=success, steps=ctrl.total_steps,
        sim_time_s=ctrl.total_steps / CONTROL_FREQ, wall_time_s=wall,
        fail_object=ctrl.fail_object, fail_state=ctrl.fail_state,
        fail_reason=ctrl.fail_reason, objects=rows,
        obj_steps=dict(ctrl.obj_steps),
    )


def stats(x):
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return "n/a"
    return (f"mean {x.mean():.2f} | std {x.std():.2f} | median {np.median(x):.2f} "
            f"| p95 {np.percentile(x, 95):.2f} | max {x.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed")
    ap.add_argument("--fixed-layout", action="store_true",
                    help="objects in the fixed row instead of random positions")
    ap.add_argument("--out", type=str, default="benchmark_results",
                    help="output prefix (writes <out>.csv and <out>.json)")
    args = ap.parse_args()

    ce.RANDOMIZE_OBJECTS = not args.fixed_layout

    env = make_env(seed=args.seed)
    results = []
    t_start = time.time()

    import contextlib
    import io
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            # controller state messages are hidden; one summary line per episode
            with contextlib.redirect_stdout(io.StringIO()):
                r = run_episode(env, seed)
            results.append(r)
            tag = "OK  " if r["success"] else "FAIL"
            extra = "" if r["success"] else f"  <- {r['fail_object']} / {r['fail_state']}: {r['fail_reason']}"
            print(f"[{i + 1:>3}/{args.episodes}] seed={seed:<6} {tag} "
                  f"steps={r['steps']:<5}{extra}", flush=True)
    finally:
        env.close()

    n = len(results)
    k = sum(r["success"] for r in results)
    lo, hi = wilson_ci(k, n)

    print("\n" + "=" * 64)
    print(f"BENCHMARK  |  Franka Panda, 3-object pick & place into Frame_B")
    print(f"layout: {'fixed row' if args.fixed_layout else 'random'} | "
          f"episodes: {n} | seeds {args.seed}..{args.seed + n - 1}")
    print("=" * 64)
    print(f"Episode success rate : {k}/{n} = {100 * k / n:.1f}%  "
          f"(95% CI {100 * lo:.1f}% - {100 * hi:.1f}%)")

    print("\nPer-object success (object ends correctly in its slot):")
    for name in OBJECT_NAMES:
        ok = sum(r["objects"][name]["ok"] for r in results)
        print(f"  {name}: {ok}/{n} = {100 * ok / n:.1f}%")
    tot_ok = sum(r["objects"][nm]["ok"] for r in results for nm in OBJECT_NAMES)
    print(f"  all objects: {tot_ok}/{3 * n} = {100 * tot_ok / (3 * n):.1f}%")

    placed = [r["objects"][nm] for r in results for nm in OBJECT_NAMES
              if r["objects"][nm]["slot"] is not None]
    print("\nPlacement accuracy (all placed objects, after settling):")
    print(f"  xy error  [cm]  : {stats([p['xy_cm'] for p in placed])}")
    print(f"  yaw error [deg] : {stats([p['yaw_deg'] for p in placed])}")

    ok_eps = [r for r in results if r["success"]]
    print("\nSpeed (successful episodes):")
    print(f"  sim time per episode [s]: {stats([r['sim_time_s'] for r in ok_eps])}")
    per_obj = [s for r in ok_eps for s in r["obj_steps"].values()]
    print(f"  cycle time per object [s]: "
          f"{stats([s / CONTROL_FREQ for s in per_obj])}")
    print(f"  wall time per episode [s]: {stats([r['wall_time_s'] for r in results])}")

    fails = [r for r in results if not r["success"]]
    print(f"\nFailures: {len(fails)}")
    if fails:
        by_reason = Counter((r["fail_object"], r["fail_state"], r["fail_reason"])
                            for r in fails)
        for (obj, st, why), c in by_reason.most_common():
            print(f"  {c:>3} x  {obj} @ {st}: {why}")
    print(f"\nTotal wall time: {time.time() - t_start:.0f} s")

    # ---- save raw results ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(dict(args=vars(args), success=k, episodes=n,
                       ci95=[lo, hi], results=results), f, indent=2,
                  default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o) if isinstance(o, np.floating) else str(o))
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "success", "steps", "sim_time_s", "fail_object",
                    "fail_state", "fail_reason"]
                   + [f"{nm}_{m}" for nm in OBJECT_NAMES
                      for m in ("ok", "slot", "xy_cm", "yaw_deg")])
        for r in results:
            row = [r["seed"], int(r["success"]), r["steps"],
                   f"{r['sim_time_s']:.2f}", r["fail_object"], r["fail_state"],
                   r["fail_reason"]]
            for nm in OBJECT_NAMES:
                o = r["objects"][nm]
                row += [int(o["ok"]), o["slot"], f"{o['xy_cm']:.3f}",
                        f"{o['yaw_deg']:.2f}"]
            w.writerow(row)
    print(f"Saved: {args.out}.csv, {args.out}.json")


if __name__ == "__main__":
    main()
