"""
Run the scripted Panda pick-and-place without training an RL model.

Task: pick Obj_A, Obj_C, Obj_D one by one from Pickup_Zone_A and place each
inside Frame_B, perpendicular (90 deg) to Frame_B's primary axis, in
sequential non-overlapping slots. Uses PickPlaceController from custom_env2.py.
"""

import argparse
import time

import numpy as np

from custom_env4 import PickPlaceController, make_env


def run_episode(env, sleep=0.0, seed=None):
    raw_obs = env.reset_raw(seed=seed)
    controller = PickPlaceController(env)
    controller.reset(raw_obs)

    while not controller.done:
        action = controller.step(raw_obs)
        raw_obs, _, sim_done, _ = env.step_raw(action)

        if sleep > 0:
            time.sleep(sleep)

        if sim_done and not controller.done:
            print("[sim] Environment terminated before controller finished.")
            controller._announce(controller.FAILED)

    final_state = env.get_raw_state(raw_obs)
    return controller.success, final_state, controller.total_steps, controller.state


def print_placements(env, controller_results=None):
    for idx, name in sorted(env.occupied.items()):
        xy, z, yaw = env.placement_error(name, env.slots[idx])
        print(f"    {name} -> slot {idx}: xy error={xy * 100:.2f} cm, "
              f"yaw error={np.degrees(yaw):.1f} deg (target 90 deg)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = make_env(render=args.render, seed=args.seed)
    successes = 0

    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            ok, state, steps, final_state = run_episode(
                env, sleep=args.sleep, seed=seed
            )
            successes += int(ok)

            print(f"Episode {ep + 1} | success={ok} | steps={steps} | "
                  f"final state={final_state}")
            print_placements(env)
    finally:
        env.close()

    print(f"Success rate: {successes}/{args.episodes}")
    print("No neural-network training was performed.")


if __name__ == "__main__":
    main()
