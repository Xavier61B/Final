from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.timeline.ITimeline import get_current_time
from omni.isaac.core.utils.viewports import set_camera_view

from trajectories import LinearTrajectory, CircularTrajectory

import gym
from gym import spaces
import numpy as np
import torch
import math

class SawyerTask(BaseTask):
    def __init__(
        self,
        name,
        offset=None
    ) -> None:

        # task-specific parameters
        self._sawyer_position = [0.0, 0.0, 0.0]
        self._reset_dist = 3.0
        self._max_torque = 10

        # values used for defining RL buffers
        self._num_observations = 24
        self._num_actions = 7
        self._device = "cpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # logistic smoothing kernel sensitiity parameter and relevance parameter for position vs velocity tracking
        self.l = 0.5
        self.w = 0.75

        # reset max joint velocity
        self._reset_vel = 1.328

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Sawyer USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/RethinkRobotics/sawyer_instanceable.usd"

        # add the Sawyer USD to our stage
        create_prim(prim_path="/World/Sawyer", prim_type="Xform", position=self._sawyer_position)
        add_reference_to_stage(usd_path, "/World/Sawyer")
        # create an ArticulationView wrapper for our sawyer - this can be extended towards accessing multiple sawyers
        self._sawyer = ArticulationView(prim_paths_expr="/World/Sawyer*", name="sawyer_view")
        # add Sawyer ArticulationView and ground plane to the Scene
        scene.add(self._sawyer)
        scene.add_default_ground_plane()

        self._hand = scene.get_object("right_l6")
        self._sawyer.set_enabled_self_collisions(False)

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):

        self._joint_indices = torch.zeros(7)

        para = "right_j"
        for i in range(7):
            tpara = para + str(i)
            self._joint_indices[i] = self._sawyer.get_dof_index(tpara)

        # randomize all envs
        indices = torch.arange(self._sawyer.count, dtype=torch.int64, device=self._device)
        self.reset(indices)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._sawyer.num_dof), device=self._device)

        for i in range(7):
            jo = self._joint_indices[i]
            upper = self._sawyer.get_dof_limits()[:, jo, 1]
            lower = self._sawyer.get_dof_limits()[:, jo, 0]
            dof_pos[:, jo] = (upper - lower) * torch.rand(num_resets, device=self._device) + lower

        # zero all DOF velocities
        dof_vel = torch.zeros((num_resets, self._sawyer.num_dof), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._sawyer.set_joint_positions(dof_pos, indices=indices)
        self._sawyer.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0

        self.start_time = get_current_time()

        x = np.random.rand(2,1) * 20 - 10
        y = np.random.rand(2,1) * 15 + 5
        z = np.random.rand(2,1) * 12 + 3

        pt = np.hstack((x, y ,z))
        self.trajectory = LinearTrajectory(pt[0], pt[1], 5)

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)

        forces = torch.zeros((self._sawyer.count, self._sawyer.num_dof), dtype=torch.float32, device=self._device)

        self._max_torque = self._sawyer.get_max_efforts(joint_indices=torch.tensor([self._j0_dof_idx, self._j1_dof_idx, self._j2_dof_idx, self._j3_dof_idx, self._j4_dof_idx, self._j5_dof_idx, self._j6_dof_idx]))

        # add indexes to max_torque based on correct joints
        for i in range(7):
            jo = self._joint_indices[i]
            forces[:, jo] = self._max_torque[i] * actions[i]

        indices = torch.arange(self._sawyer.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self):
        dof_pos = self._sawyer.get_joint_positions()
        dof_vel = self._sawyer.get_joint_velocities()
        hand_pos = self._hand.get_world_pose()
        hand_vel = self._hand.get_linear_velocity()

        # collect pole and cart joint positions and velocities for observation
        j0_pos = dof_pos[:, self._j0_dof_idx]
        j0_vel = dof_vel[:, self._j0_dof_idx]
        j1_pos = dof_pos[:, self._j1_dof_idx]
        j1_vel = dof_vel[:, self._j1_dof_idx]
        j2_pos = dof_pos[:, self._j2_dof_idx]
        j2_vel = dof_vel[:, self._j2_dof_idx]
        j3_pos = dof_pos[:, self._j3_dof_idx]
        j3_vel = dof_vel[:, self._j3_dof_idx]
        j4_pos = dof_pos[:, self._j4_dof_idx]
        j4_vel = dof_vel[:, self._j4_dof_idx]
        j5_pos = dof_pos[:, self._j5_dof_idx]
        j5_vel = dof_vel[:, self._j5_dof_idx]
        j6_pos = dof_pos[:, self._j6_dof_idx]
        j6_vel = dof_vel[:, self._j6_dof_idx]

        self.obs[:, 0] = j0_pos
        self.obs[:, 1] = j0_vel
        self.obs[:, 2] = j1_pos
        self.obs[:, 3] = j1_vel
        self.obs[:, 4] = j2_pos
        self.obs[:, 5] = j2_vel
        self.obs[:, 6] = j3_pos
        self.obs[:, 7] = j3_vel
        self.obs[:, 8] = j4_pos
        self.obs[:, 9] = j4_vel
        self.obs[:, 10] = j5_pos
        self.obs[:, 11] = j5_vel
        self.obs[:, 12] = j6_pos
        self.obs[:, 13] = j6_vel
        self.obs[:, 14] = hand_pos[1][0]
        self.obs[:, 15] = hand_pos[1][1]
        self.obs[:, 16] = hand_pos[1][2]
        self.obs[:, 17] = hand_pos[1][3]
        self.obs[:, 18] = hand_pos[0][0]
        self.obs[:, 19] = hand_pos[0][1]
        self.obs[:, 20] = hand_pos[0][2]
        self.obs[:, 21] = hand_vel[0]
        self.obs[:, 22] = hand_vel[1]
        self.obs[:, 23] = hand_vel[2]

        return self.obs

    def calculate_metrics(self) -> None:
        j_vel = torch.zeros(7)
        j_pos = torch.zeros(7)

        j_pos[0] = self.obs[:, 0]
        j_vel[0] = self.obs[:, 1]
        j_pos[1] = self.obs[:, 2]
        j_vel[1] = self.obs[:, 3]
        j_pos[2] = self.obs[:, 4]
        j_vel[2] = self.obs[:, 5]
        j_pos[3] = self.obs[:, 6]
        j_vel[3] = self.obs[:, 7]
        j_pos[4] = self.obs[:, 8]
        j_vel[4] = self.obs[:, 9]
        j_pos[5] = self.obs[:, 10]
        j_vel[5] = self.obs[:, 11]
        j_pos[6] = self.obs[:, 12]
        j_vel[6] = self.obs[:, 13]

        hand_pos = torch.zeros(7)

        hand_pos[0] = self.obs[:, 14]
        hand_pos[1] = self.obs[:, 15]
        hand_pos[2] = self.obs[:, 16]
        hand_pos[3] = self.obs[:, 17]
        hand_pos[4] = self.obs[:, 18]
        hand_pos[5] = self.obs[:, 19]
        hand_pos[6] = self.obs[:, 20]

        hand_vel = torch.zeros(3)

        hand_vel[0] = self.obs[:, 21]
        hand_vel[1] = self.obs[:, 22] 
        hand_vel[2] = self.obs[:, 23]

        # compute penalty based on joint velocities
        time = get_current_time() - self.start_time

        # get trajectories and errors
        pd = self.trajectory.target_pose(time)
        vd = self.trajectory.target_velocity(time)

        pe = hand_pos - pd
        ve = hand_vel - vd

        # compute reward based on gripper pose and position vs trajectory, use get_current_time() and self.start_time
        klog = lambda x, l: 2 / (torch.exp(x * l) + torch.exp(-x * l))
        reward = self.w * klog(pe, self.l) + (1 - self.w) * klog(ve, self.l)

        # compute penalty if dof limits near exceeded, use get_dof_limits
        reward = torch.where(torch.max(torch.abs(j_vel)) > torch.ones_like(reward) * -2.0, 1, reward)


        # compute penalty if jacobian is near singular, use get_jacobians

        # compute reward based on angle of pole and cart velocity
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # apply a penalty if cart is too far from center
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # apply a penalty if pole is too far from upright
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        return reward.item()

    def is_done(self) -> None:
        j_vel = torch.zeros(7)
        j_pos = torch.zeros(7)

        j_pos[0] = self.obs[:, 0]
        j_vel[0] = self.obs[:, 1]
        j_pos[1] = self.obs[:, 2]
        j_vel[1] = self.obs[:, 3]
        j_pos[2] = self.obs[:, 4]
        j_vel[2] = self.obs[:, 5]
        j_pos[3] = self.obs[:, 6]
        j_vel[3] = self.obs[:, 7]
        j_pos[4] = self.obs[:, 8]
        j_vel[4] = self.obs[:, 9]
        j_pos[5] = self.obs[:, 10]
        j_vel[5] = self.obs[:, 11]
        j_pos[6] = self.obs[:, 12]
        j_vel[6] = self.obs[:, 13]

        limits = self._sawyer.get_dof_limits()

        # reset if sawyer joint velocities too high, dof limits exceeded, or trajectory finished
        resets = torch.where(torch.max(torch.abs(j_vel)) > self._reset_vel, 1, 0)
        torch.where()

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        self.resets = resets

        return resets.item()
