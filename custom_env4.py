"""
Scripted multi-object pick-and-place for a Franka Panda in robosuite.

Task: pick Obj_A, Obj_C, Obj_D one by one from Pickup_Zone_A and place each
one inside Frame_B, with its long axis PERPENDICULAR (90 deg) to the primary
axis of Frame_B. Objects are put in sequential, non-overlapping slots.

No RL model is trained or loaded. The robot uses robosuite's default Panda
OSC_POSE controller and a state machine.

Action sent to robosuite: [dx, dy, dz, droll, dpitch, dyaw, gripper]
(position, yaw and gripper are used; roll/pitch deltas stay at zero).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_geom


CONTROL_FREQ = 20
MAX_STEPS = 2400  # budget for all three objects

MAX_POS_DELTA = 0.045   # m, command saturation (OSC output range is 0.05)
MAX_YAW_DELTA = 0.5     # rad, OSC rotation output range

APPROACH_HEIGHT = 0.12
LIFT_HEIGHT = 0.14
POSITION_TOL = 0.012
GRASP_TOL = 0.012
YAW_TOL = np.deg2rad(4.0)
TILT_GAIN = 1.0

PLACE_XY_TOL = 0.025
PLACE_Z_TOL = 0.020
PLACE_YAW_TOL = np.deg2rad(15.0)

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# ---------------------------------------------------------------- geometry
OBJECT_NAMES = ["Obj_A", "Obj_C", "Obj_D"]
OBJECT_COLORS = {
    "Obj_A": (0.9, 0.1, 0.1, 1.0),
    "Obj_C": (0.1, 0.3, 0.9, 1.0),
    "Obj_D": (0.95, 0.75, 0.1, 1.0),
}
# half sizes: long axis = local x (7 cm), short axis = local y (4 cm)
OBJ_HALF = np.array([0.035, 0.020, 0.020])

PICKUP_CENTER = np.array([0.0, -0.20])   # Pickup_Zone_A
PICKUP_X_OFFSETS = [-0.11, 0.0, 0.11]
PICKUP_JITTER_XY = 0.01
PICKUP_YAW_RANGE = np.deg2rad(15.0)

# Random placement of the three objects on the table (reachable region that
# does not overlap Frame_B). Set RANDOMIZE_OBJECTS=False for the fixed row.
RANDOMIZE_OBJECTS = True
RANDOM_X_RANGE = (-0.15, 0.15)
RANDOM_Y_RANGE = (-0.30, -0.08)
RANDOM_YAW_RANGE = np.deg2rad(90.0)   # +/- 90 deg = any orientation
MIN_OBJ_DISTANCE = 0.11               # keeps room for the open fingers

FRAME_CENTER = np.array([0.0, 0.20])     # Frame_B
FRAME_AXIS_YAW = 0.0                     # primary axis of Frame_B = world +x
FRAME_SIZE = (0.32, 0.11)                # (length along axis, width)
SLOT_SPACING = 0.09
N_SLOTS = 3

PERPENDICULAR_YAW = np.pi / 2            # relative to Frame_B primary axis


def wrap_pi(a):
    """Wrap angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def wrap_half_pi(a):
    """Wrap angle to (-pi/2, pi/2] (a box looks the same after 180 deg)."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def choose_grasp_yaw(obj_yaw, eef_yaw, limit=np.deg2rad(100.0)):
    """Gripper yaw that matches a box (a box looks the same every 180 deg).

    Among the equivalent yaws inside +/-`limit` (keeps the wrist away from its
    joint limit) pick the one closest to the current gripper yaw, so the wrist
    never has to spin ~180 deg when the box sits near the +/-90 deg boundary."""
    base = wrap_half_pi(obj_yaw)
    cands = [c for c in (base, base - np.pi, base + np.pi) if abs(c) <= limit]
    return float(min(cands, key=lambda c: abs(c - eef_yaw)))


def mat_yaw(mat, col):
    """Yaw (about world z) of the given column axis of a rotation matrix."""
    return float(np.arctan2(mat[1, col], mat[0, col]))


# ------------------------------------------------------------ robosuite env
class MultiObjectLift(Lift):
    """Lift table scene with 3 boxes in Pickup_Zone_A and a drawn Frame_B."""

    def __init__(self, *args, show_target=True, **kwargs):
        self._show_target = bool(show_target)
        self.objs = {}
        self.obj_body_ids = {}
        super().__init__(*args, **kwargs)

    # ---- model ----
    def _load_model(self):
        # Skip Lift._load_model (single cube); call grandparent for the robot.
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.objs = {}
        for name in OBJECT_NAMES:
            self.objs[name] = BoxObject(
                name=name,
                size=list(OBJ_HALF),
                rgba=list(OBJECT_COLORS[name]),
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=list(self.objs.values()),
        )

        if self._show_target:
            self._draw_frame_b()

    def _draw_frame_b(self):
        """Visual-only outline of Frame_B plus its slot markers."""
        zt = self.table_offset[2] + 0.001
        L, W = FRAME_SIZE
        t = 0.004  # wall half thickness
        c = np.array([FRAME_CENTER[0], FRAME_CENTER[1]])
        vis = dict(contype=0, conaffinity=0, group=1, rgba=(0.1, 0.8, 0.2, 0.9))
        walls = [
            ("fb_n", (L / 2, t, 0.001), (0, W / 2)),
            ("fb_s", (L / 2, t, 0.001), (0, -W / 2)),
            ("fb_e", (t, W / 2, 0.001), (L / 2, 0)),
            ("fb_w", (t, W / 2, 0.001), (-L / 2, 0)),
        ]
        for name, size, off in walls:
            self.model.worldbody.append(
                new_geom(name=name, type="box", size=size,
                         pos=(c[0] + off[0], c[1] + off[1], zt), **vis)
            )
        for i, s in enumerate(self.slot_positions()):
            self.model.worldbody.append(
                new_geom(
                    name=f"slot_{i}", type="box",
                    size=(OBJ_HALF[1], OBJ_HALF[0], 0.0005),  # perpendicular
                    pos=(s[0], s[1], zt),
                    contype=0, conaffinity=0, group=1, rgba=(0.2, 0.4, 1.0, 0.35),
                )
            )

    @staticmethod
    def slot_positions():
        """Slot centers along Frame_B's primary axis (world xy)."""
        ax = np.array([np.cos(FRAME_AXIS_YAW), np.sin(FRAME_AXIS_YAW)])
        offs = (np.arange(N_SLOTS) - (N_SLOTS - 1) / 2) * SLOT_SPACING
        return [FRAME_CENTER + o * ax for o in offs]

    # ---- references / observables ----
    def _setup_references(self):
        super(Lift, self)._setup_references()
        self.obj_body_ids = {
            n: self.sim.model.body_name2id(o.root_body)
            for n, o in self.objs.items()
        }

    def _setup_observables(self):
        # Robot observables only; objects are read directly from the sim.
        return super(Lift, self)._setup_observables()

    # ---- reset ----
    def _reset_internal(self):
        super(Lift, self)._reset_internal()
        z = self.table_offset[2] + OBJ_HALF[2] + 0.001
        for name, (x, y, yaw) in zip(OBJECT_NAMES, self._sample_object_poses()):
            quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # wxyz
            self.sim.data.set_joint_qpos(
                self.objs[name].joints[0],
                np.concatenate([[x, y, z], quat]),
            )
        self.sim.forward()

    def _sample_object_poses(self):
        """Return [(x, y, yaw)] for each object, no two closer than
        MIN_OBJ_DISTANCE (rejection sampling)."""
        if not RANDOMIZE_OBJECTS:
            poses = []
            for dx in PICKUP_X_OFFSETS:
                poses.append((
                    PICKUP_CENTER[0] + dx + self.rng.uniform(
                        -PICKUP_JITTER_XY, PICKUP_JITTER_XY),
                    PICKUP_CENTER[1] + self.rng.uniform(
                        -PICKUP_JITTER_XY, PICKUP_JITTER_XY),
                    self.rng.uniform(-PICKUP_YAW_RANGE, PICKUP_YAW_RANGE),
                ))
            return poses

        for _ in range(1000):
            poses = []
            for _ in OBJECT_NAMES:
                for _ in range(100):
                    xy = np.array([self.rng.uniform(*RANDOM_X_RANGE),
                                   self.rng.uniform(*RANDOM_Y_RANGE)])
                    if all(np.linalg.norm(xy - np.array(p[:2])) >= MIN_OBJ_DISTANCE
                           for p in poses):
                        poses.append((xy[0], xy[1], self.rng.uniform(
                            -RANDOM_YAW_RANGE, RANDOM_YAW_RANGE)))
                        break
                else:
                    break
            if len(poses) == len(OBJECT_NAMES):
                return poses
        raise RuntimeError("Could not place objects without overlap")

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    # ---- helpers ----
    def obj_pose(self, name):
        bid = self.obj_body_ids[name]
        pos = np.array(self.sim.data.body_xpos[bid], dtype=np.float64)
        mat = np.array(self.sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
        return pos, mat


# ------------------------------------------------------ gym-style wrapper
class PickPlaceEnv(gym.Env):
    """Panda + 3 objects + Frame_B, with a thin Gymnasium-compatible wrapper."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False, offscreen=False, seed=None):
        super().__init__()
        self.render_on = bool(render)
        self.offscreen_on = bool(offscreen)

        self.env = MultiObjectLift(
            robots="Panda",
            show_target=True,
            has_renderer=self.render_on,
            has_offscreen_renderer=self.offscreen_on,
            use_camera_obs=False,
            use_object_obs=False,
            reward_shaping=False,
            control_freq=CONTROL_FREQ,
            horizon=MAX_STEPS,
            seed=seed,
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        self.object_names = list(OBJECT_NAMES)
        self.slots = [np.append(s, 0.0) for s in MultiObjectLift.slot_positions()]
        self.occupied = {}            # slot index -> object name
        self.cube_start = {}          # name -> initial position
        self.obj_z0 = 0.0
        self.steps = 0

        obs = self._reset_sim()
        flat = self._flatten(obs)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=flat.shape, dtype=np.float32)

    # ------------------------------------------------------------- slots
    def get_next_available_slot(self, frame_id="Frame_B",
                                orientation="perpendicular"):
        """Return (xyz, yaw_degrees_relative_to_frame, slot_index) of the first
        free slot, in order along Frame_B's primary axis."""
        for i, s in enumerate(self.slots):
            if i not in self.occupied:
                yaw = 90.0 if orientation == "perpendicular" else 0.0
                return s.copy(), yaw, i
        raise RuntimeError("No free slot left in Frame_B")

    def reserve_slot(self, idx, name):
        self.occupied[idx] = name

    # ------------------------------------------------------------- reset
    def _reset_sim(self):
        obs = self.env.reset()
        self.occupied = {}
        self.cube_start = {n: self.env.obj_pose(n)[0].copy()
                           for n in self.object_names}
        self.obj_z0 = float(self.cube_start[self.object_names[0]][2])
        for s in self.slots:
            s[2] = self.obj_z0
        self.steps = 0
        return obs

    def _flatten(self, obs):
        parts = [obs["robot0_proprio-state"]]
        for n in self.object_names:
            pos, mat = self.env.obj_pose(n)
            parts += [pos, [mat_yaw(mat, 0)]]
        return np.concatenate(
            [np.asarray(p, dtype=np.float64).ravel() for p in parts]
        ).astype(np.float32)

    # --------------------------------------------------------- state info
    def _is_grasped(self, name):
        gripper = self.env.robots[0].gripper
        if isinstance(gripper, dict):
            gripper = next(iter(gripper.values()))
        return bool(self.env._check_grasp(
            gripper=gripper, object_geoms=self.env.objs[name]))

    def eef_yaw(self, obs):
        from robosuite.utils import transform_utils as T
        mat = T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
        return mat_yaw(mat, 0)

    def eef_tilt_error(self, obs):
        """Rotation (axis*angle, world xy) that points the gripper straight
        down again. OSC rotation deltas are expressed in the base frame."""
        from robosuite.utils import transform_utils as T
        mat = T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
        z_axis = mat[:, 2]
        err = np.cross(z_axis, np.array([0.0, 0.0, -1.0]))
        return err[:2]

    def get_raw_state(self, obs, name=None):
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        state = {"eef": eef, "eef_yaw": self.eef_yaw(obs), "objects": {}}
        for n in self.object_names:
            pos, mat = self.env.obj_pose(n)
            state["objects"][n] = {
                "pos": pos,
                "yaw": mat_yaw(mat, 0),        # yaw of the long axis
                "grasped": self._is_grasped(n),
            }
        if name is not None:
            state.update(state["objects"][name])
            state["cube"] = state["pos"]
        return state

    def placement_error(self, name, slot_xyz, rel_yaw=PERPENDICULAR_YAW):
        """xy / z / yaw error of object `name` w.r.t. a slot.
        Yaw error is measured on the long axis, modulo 180 degrees."""
        pos, mat = self.env.obj_pose(name)
        want = FRAME_AXIS_YAW + rel_yaw
        yaw_err = abs(wrap_half_pi(mat_yaw(mat, 0) - want))
        return (float(np.linalg.norm(pos[:2] - slot_xyz[:2])),
                float(abs(pos[2] - slot_xyz[2])),
                float(yaw_err))

    # ------------------------------------------------------------ actions
    def make_action(self, position_cmd, gripper, yaw_cmd=0.0):
        pos = np.asarray(position_cmd, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError("position_cmd must have shape (3,)")
        dim = int(self.env.action_dim)
        if dim != 7:
            raise RuntimeError(
                f"Unexpected Panda action dimension: {dim}. Expected 7 (OSC_POSE).")
        a = np.zeros(7, dtype=np.float64)
        a[:3] = np.clip(pos, -1.0, 1.0)
        a[5] = float(np.clip(yaw_cmd, -1.0, 1.0))
        a[6] = float(np.clip(gripper, -1.0, 1.0))
        return a

    def reset_raw(self, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self._reset_sim()

    def step_raw(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape == (7,):          # full OSC_POSE action, passed through
            pass
        elif a.shape == (5,):        # [dx, dy, dz, dyaw, gripper]
            a = self.make_action(a[:3], a[4], a[3])
        elif a.shape == (4,):        # [dx, dy, dz, gripper]
            a = self.make_action(a[:3], a[3])

        obs, reward, done, info = self.env.step(a)
        self.steps += 1
        if self.render_on:
            self.env.render()

        info = dict(info or {})
        info["placed"] = {
            idx: self.placement_error(n, self.slots[idx])
            for idx, n in self.occupied.items()
        }
        return obs, float(reward), bool(done), info

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs = self._reset_sim()
        return self._flatten(obs), {}

    def step(self, action):
        obs, reward, done, info = self.step_raw(action)
        ok = len(self.occupied) == len(self.object_names) and all(
            e[0] < PLACE_XY_TOL and e[1] < PLACE_Z_TOL and e[2] < PLACE_YAW_TOL
            for e in info["placed"].values())
        truncated = bool(self.steps >= MAX_STEPS and not ok)
        return self._flatten(obs), reward, ok, truncated, info

    def render_frame(self, camera="agentview", height=480, width=640):
        return self.env.sim.render(
            camera_name=camera, height=height, width=width)[::-1]

    def close(self):
        self.env.close()


# -------------------------------------------------------------- controller
class PickPlaceController:
    """Deterministic state machine that handles Obj_A, Obj_C, Obj_D in turn."""

    ALIGN_PICK = "ALIGN_PICK"          # move above object and match yaw
    DESCEND_PICK = "DESCEND_PICK"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    APPROACH_PLACE = "APPROACH_PLACE"  # move above slot, rotate to 90 deg
    DESCEND_PLACE = "DESCEND_PLACE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    RETRACT = "RETRACT"
    DONE = "DONE"
    FAILED = "FAILED"

    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.state = self.ALIGN_PICK
        self.state_steps = 0
        self.total_steps = 0
        self.success = False

        self.queue = list(OBJECT_NAMES)
        self.current = None
        self.results = {}            # name -> (xy_err, z_err, yaw_err, ok)
        self.obj_steps = {}          # name -> controller steps spent on it
        self._obj_start = 0
        self.fail_reason = None
        self.fail_state = None
        self.fail_object = None
        self.slot_idx = None
        self.slot = np.zeros(3)
        self.place_rel_yaw = PERPENDICULAR_YAW

        self.pick_target = np.zeros(3)
        self.pick_yaw = 0.0
        self.place_yaw = 0.0
        self.grasp_eef_yaw = 0.0
        self._tilt_cmd = np.zeros(2)

    # ----------------------------------------------------------- helpers
    def _announce(self, state):
        self.state = state
        self.state_steps = 0
        tag = f" ({self.current})" if self.current else ""
        print(f"[controller] {state}{tag}")

    def _set_state(self, state):          # kept for backward compatibility
        self._announce(state)

    def reset(self, raw_obs):
        self.queue = list(OBJECT_NAMES)
        self.results = {}
        self.obj_steps = {}
        self.fail_reason = self.fail_state = self.fail_object = None
        self.total_steps = 0
        self.success = False
        self.env.occupied = {}
        self._start_next_object(raw_obs)

    def _start_next_object(self, raw_obs):
        if not self.queue:
            self.success = all(r[3] for r in self.results.values()) and \
                len(self.results) == len(OBJECT_NAMES)
            self.current = None
            self._announce(self.DONE)
            return
        self.current = self.queue.pop(0)
        self._obj_start = self.total_steps
        s = self.env.get_raw_state(raw_obs, self.current)
        cube = s["pos"]
        self.pick_target = cube.copy()
        self.pick_target[2] += 0.005
        # Fingers close along the object's short axis (local y). The default
        # gripper finger axis is the eef y axis, so eef yaw == object yaw.
        self.pick_yaw = choose_grasp_yaw(s["yaw"], s["eef_yaw"])

        idx_xyz_yaw = self.env.get_next_available_slot(
            "Frame_B", orientation="perpendicular")
        self.slot, rel_yaw_deg, self.slot_idx = idx_xyz_yaw
        self.env.reserve_slot(self.slot_idx, self.current)
        self.place_rel_yaw = np.deg2rad(rel_yaw_deg)
        self._announce(self.ALIGN_PICK)

    @staticmethod
    def _position_action(eef, target):
        delta = np.asarray(target) - np.asarray(eef)
        return np.clip(delta / MAX_POS_DELTA, -1.0, 1.0)

    @staticmethod
    def _yaw_action(current, target):
        err = wrap_pi(target - current)
        return float(np.clip(err / MAX_YAW_DELTA * 1.5, -1.0, 1.0))

    @staticmethod
    def _arrived(eef, target, tol=POSITION_TOL):
        return float(np.linalg.norm(np.asarray(target) - np.asarray(eef))) < tol

    def _act(self, pos_cmd, yaw_cmd, grip):
        """Full OSC_POSE action. Roll/pitch deltas keep the gripper vertical
        (otherwise the held object tilts after the 90 deg yaw rotation)."""
        tilt = np.clip(self._tilt_cmd, -1.0, 1.0)
        return np.array([*pos_cmd, tilt[0], tilt[1], yaw_cmd, grip],
                        dtype=np.float32)

    def _hold(self, grip):
        return self._act([0, 0, 0], 0.0, grip)

    def _fail(self, why=""):
        if why:
            print(f"[controller] failure reason: {why}")
        self.fail_reason = why or "unknown"
        self.fail_state = self.state
        self.fail_object = self.current
        self.results[self.current] = (np.nan, np.nan, np.nan, False)
        self._announce(self.FAILED)
        return self._hold(GRIP_OPEN)

    # -------------------------------------------------------------- step
    def step(self, raw_obs):
        if self.done:
            return self._hold(GRIP_OPEN)

        self.total_steps += 1
        self.state_steps += 1
        if self.total_steps > MAX_STEPS:
            return self._fail("step budget exhausted")

        s = self.env.get_raw_state(raw_obs, self.current)
        eef, eef_yaw = s["eef"], s["eef_yaw"]
        cube = s["pos"]
        z0 = self.env.obj_z0
        self._tilt_cmd = TILT_GAIN * self.env.eef_tilt_error(raw_obs) / MAX_YAW_DELTA

        # --- 1) hover above the object, rotate gripper to match its yaw
        if self.state == self.ALIGN_PICK:
            # follow the live object pose in case it was nudged
            self.pick_target = cube.copy()
            self.pick_target[2] += 0.005
            self.pick_yaw = choose_grasp_yaw(s["yaw"], eef_yaw)
            target = self.pick_target.copy()
            target[2] = z0 + APPROACH_HEIGHT
            yaw_err = abs(wrap_pi(self.pick_yaw - eef_yaw))
            if self._arrived(eef, target) and yaw_err < YAW_TOL:
                self._announce(self.DESCEND_PICK)
                return self._hold(GRIP_OPEN)
            if self.state_steps > 150:
                return self._fail("timeout while aligning above the object")
            return self._act(self._position_action(eef, target),
                             self._yaw_action(eef_yaw, self.pick_yaw), GRIP_OPEN)

        # --- 2) straight down onto the object
        if self.state == self.DESCEND_PICK:
            if self._arrived(eef, self.pick_target, GRASP_TOL):
                self._announce(self.CLOSE_GRIPPER)
                return self._hold(GRIP_CLOSE)
            if self.state_steps > 80:
                return self._fail("timeout while descending to the object")
            return self._act(self._position_action(eef, self.pick_target),
                             self._yaw_action(eef_yaw, self.pick_yaw), GRIP_OPEN)

        # --- 3) close the fingers
        if self.state == self.CLOSE_GRIPPER:
            if s["grasped"] and self.state_steps >= 8:
                # Rotation needed so the long axis ends up perpendicular
                # to Frame_B's primary axis.
                want = FRAME_AXIS_YAW + self.place_rel_yaw
                delta = wrap_half_pi(want - s["yaw"])
                self.grasp_eef_yaw = eef_yaw
                self.place_yaw = eef_yaw + delta
                self._announce(self.LIFT)
            elif self.state_steps >= 40:
                return self._fail("gripper could not grasp the object")
            return self._hold(GRIP_CLOSE)

        # --- 4) lift straight up
        if self.state == self.LIFT:
            lift_target = np.array([eef[0], eef[1], z0 + LIFT_HEIGHT])
            if not s["grasped"] and self.state_steps > 8:
                return self._fail("object lost while lifting")
            if self._arrived(eef, lift_target, tol=0.02):
                self._announce(self.APPROACH_PLACE)
                return self._hold(GRIP_CLOSE)
            return self._act(self._position_action(eef, lift_target), 0.0, GRIP_CLOSE)

        # --- 5) carry to the slot and rotate the object by ~90 deg
        # (position is servoed on the OBJECT pose, so grasp offsets and
        # controller sag do not shift where the object ends up)
        if self.state == self.APPROACH_PLACE:
            want_cube = np.array([self.slot[0], self.slot[1], z0 + LIFT_HEIGHT])
            obj_yaw_err = wrap_half_pi(
                FRAME_AXIS_YAW + self.place_rel_yaw - s["yaw"])
            yaw_err = abs(obj_yaw_err)
            if not s["grasped"]:
                return self._fail("object dropped while carrying")
            if (np.linalg.norm(want_cube - cube) < 0.018
                    and yaw_err < YAW_TOL):
                self._announce(self.DESCEND_PLACE)
                return self._hold(GRIP_CLOSE)
            if self.state_steps > 200:
                return self._fail("timeout while carrying to Frame_B")
            return self._act(self._position_action(eef, eef + want_cube - cube),
                             self._yaw_action(0.0, obj_yaw_err), GRIP_CLOSE)

        # --- 6) lower into the slot
        if self.state == self.DESCEND_PLACE:
            want_cube = np.array([self.slot[0], self.slot[1], z0 + 0.002])
            xy_err = np.linalg.norm(want_cube[:2] - cube[:2])
            obj_yaw_err = wrap_half_pi(
                FRAME_AXIS_YAW + self.place_rel_yaw - s["yaw"])
            if (xy_err < 0.012 and cube[2] - z0 < 0.010
                    and abs(obj_yaw_err) < np.deg2rad(8.0)):
                self._announce(self.OPEN_GRIPPER)
                return self._hold(GRIP_OPEN)
            if not s["grasped"]:
                return self._fail("object slipped out of the gripper while lowering")
            if self.state_steps > 100:
                return self._fail("timeout while lowering into the slot")
            return self._act(self._position_action(eef, eef + want_cube - cube),
                             self._yaw_action(0.0, obj_yaw_err), GRIP_CLOSE)

        # --- 7) release
        if self.state == self.OPEN_GRIPPER:
            if self.state_steps >= 12 and not s["grasped"]:
                self._announce(self.RETRACT)
            elif self.state_steps > 60:
                return self._fail("gripper did not release the object")
            return self._hold(GRIP_OPEN)

        # --- 8) retract, verify, move on to the next object
        if self.state == self.RETRACT:
            retract = np.array([self.slot[0], self.slot[1], z0 + LIFT_HEIGHT])
            if self._arrived(eef, retract, tol=0.02):
                xy, z, yaw = self.env.placement_error(
                    self.current, self.slot, self.place_rel_yaw)
                ok = (xy < PLACE_XY_TOL and z < PLACE_Z_TOL
                      and yaw < PLACE_YAW_TOL and not s["grasped"])
                self.results[self.current] = (xy, z, yaw, ok)
                self.obj_steps[self.current] = self.total_steps - self._obj_start
                print(f"[controller] {self.current} -> slot {self.slot_idx}: "
                      f"xy={xy*100:.1f} cm, yaw err={np.degrees(yaw):.1f} deg, "
                      f"ok={ok}")
                if not ok:
                    return self._fail("placement outside tolerance")
                self._start_next_object(raw_obs)
                return self._hold(GRIP_OPEN)
            if self.state_steps > 60:
                return self._fail("timeout while retracting")
            return self._act(self._position_action(eef, retract), 0.0, GRIP_OPEN)

        return self._hold(GRIP_OPEN)

    @property
    def done(self):
        return self.state in {self.DONE, self.FAILED}

    @property
    def ok(self):
        return self.state == self.DONE and self.success


def make_env(render=False, offscreen=False, seed=None):
    return PickPlaceEnv(render=render, offscreen=offscreen, seed=seed)