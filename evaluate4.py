"""
Visual evaluation for the scripted Panda pick-and-place controller.

Picks Obj_A, Obj_C, Obj_D one by one and places each in Frame_B at 90 deg to
the frame's primary axis. No model file, SAC, PyTorch or VecNormalize needed.
"""

import argparse
import time

import numpy as np

from custom_env4 import PickPlaceController, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    env = make_env(render=True, seed=args.seed)
    successes = 0

    try:
        for ep in range(args.episodes):
            raw_obs = env.reset_raw(seed=args.seed + ep)
            controller = PickPlaceController(env)
            controller.reset(raw_obs)

            print(f"\nEpisode {ep + 1}:")
            for name, pos in env.cube_start.items():
                print(f"  {name} at A={pos.round(3)}")

            while not controller.done:
                action = controller.step(raw_obs)
                raw_obs, _, sim_done, info = env.step_raw(action)

                if args.sleep > 0:
                    time.sleep(args.sleep)

                if sim_done and not controller.done:
                    controller._announce(controller.FAILED)

            ok = bool(controller.success)
            successes += int(ok)

            print(f"Episode {ep + 1} | success={ok} | steps={controller.total_steps}")
            for idx, name in sorted(env.occupied.items()):
                xy, z, yaw = env.placement_error(name, env.slots[idx])
                print(f"  {name} -> slot {idx}: xy error={xy * 100:.2f} cm, "
                      f"yaw error={np.degrees(yaw):.1f} deg (target 90 deg)")

        print(f"\nSuccess rate: {successes}/{args.episodes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
