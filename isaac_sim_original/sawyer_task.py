from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.prims import RigidPrim
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

        # values used for defining RL buffers
        self._num_observations = 34
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
        self.l1 = 4
        self.l2 = 1
        self.w = 0.6

        # reset max joint velocity
        self._reset_vel = 1.328 # 1.328
        self.pre_pos = torch.zeros(3)
        self.first_time = True
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
        self._hand = RigidPrim(prim_path="/World/Sawyer/right_hand", name="right_hand_cap")
        self.simulation = SimulationContext()
        self.simulation.set_simulation_dt(1/100)
        print(self.simulation.get_physics_dt())
        # add Sawyer ArticulationView and ground plane to the Scene
        scene.add(self._sawyer)
        scene.add(self._hand)
        scene.add_default_ground_plane()
        self._sawyer.set_enabled_self_collisions(torch.zeros(self._sawyer.count, dtype=torch.bool))


        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[3, 3, 1], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        self._joint_indices = torch.zeros(7)

        para = "right_j"
        for i in range(7):
            tpara = para + str(i)
            self._joint_indices[i] = self._sawyer.get_dof_index(tpara)

        self._joint_indices = self._joint_indices.long()

        # randomize all envs
        indices = torch.arange(self._sawyer.count, dtype=torch.int64, device=self._device)
        self.reset(indices)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._sawyer.num_dof), device=self._device)

        tuck = [0, -1, 0, 1, 0, 1.6, np.pi/2]
        for i in range(7):
            jo = int(self._joint_indices[i].item())
            upper = self._sawyer.get_dof_limits()[:, jo, 1]
            lower = self._sawyer.get_dof_limits()[:, jo, 0]
            dof_pos[:, jo] = torch.tensor([tuck[i]]) #(upper - lower) * torch.rand(num_resets, device=self._device) + lower

        # zero all DOF velocities
        dof_vel = torch.zeros((num_resets, self._sawyer.num_dof), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._sawyer.set_joint_velocities(dof_vel, indices=indices)
        self._sawyer.set_joint_positions(dof_pos, indices=indices)

        hand_pos = self._hand.get_world_pose()[0].numpy()
        # bookkeeping
        self.resets[env_ids] = 0
        self.start_time = self.simulation.current_time
        x = np.random.rand(2,1) - 0.5
        y = np.random.rand(2,1) - 0.5
        z = np.random.rand(2,1) * 0.5 + 0.25

        pt = np.hstack((x, y ,z))
        self.trajectory = LinearTrajectory(hand_pos, hand_pos + np.array([0, 1, 0]), 5)

    def pre_physics_step(self, actions) -> None:
        time = self.simulation.current_time - self.start_time
        hand_pos = self._hand.get_world_pose()
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            reset_env_ids = torch.tensor([0])
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)
        forces = torch.zeros((self._sawyer.count, self._sawyer.num_dof-1), dtype=torch.float32, device=self._device)

        self._max_torque = self._sawyer.get_max_efforts(joint_indices=self._joint_indices)
        #force_multiplier = torch.tensor([0.16,0.08,0.04,0.02,0.01,0.005,0.0025])
        # add indexes to max_torque based on correct joints
        for i in range(7):
            forces[:, i] = self._max_torque[0][i] * actions[i] #* force_multiplier[i]

        indices = torch.arange(self._sawyer.count, dtype=torch.int32, device=self._device)
        self._sawyer.set_joint_efforts(forces, joint_indices=self._joint_indices)

    def get_observations(self):
        dof_pos = self._sawyer.get_joint_positions()
        dof_pos = dof_pos[:,self._joint_indices]
        
        dof_vel = self._sawyer.get_joint_velocities()
        dof_vel = dof_vel[:,self._joint_indices]

        hand_pos = self._hand.get_world_pose()
        time = self.simulation.current_time - self.start_time

        #TODO: replace with actual dt value
        hand_vel = (hand_pos[0] - self.pre_pos) / (1/100)
        
        if (self.first_time == True):
            hand_vel = torch.zeros(3)
            self.first_time = False

        self.pre_pos = hand_pos[0]

        # collect pole and cart joint positions and velocities for observation
        j_vel = torch.zeros(7)
        j_pos = torch.zeros(7)
        for i in range(7):
            j_pos[i] = dof_pos[:, i]
            j_vel[i] = dof_vel[:, i]

        for i in range(7):
            ind = 2 * i
            self.obs[:, ind] = j_pos[i]
            self.obs[:, ind+1] = j_vel[i]
        pd = self.trajectory.target_pose(time)
        vd = self.trajectory.target_velocity(time)
        self.obs[:, 14] = hand_pos[0][0]
        self.obs[:, 15] = hand_pos[0][1]
        self.obs[:, 16] = hand_pos[0][2]
        self.obs[:, 17] = hand_pos[1][0]
        self.obs[:, 18] = hand_pos[1][1]
        self.obs[:, 19] = hand_pos[1][2]
        self.obs[:, 20] = hand_pos[1][3]
        self.obs[:, 21] = hand_vel[0]
        self.obs[:, 22] = hand_vel[1]
        self.obs[:, 23] = hand_vel[2]
        self.obs[:, 24] = pd[0]
        self.obs[:, 25] = pd[1]
        self.obs[:, 26] = pd[2]
        self.obs[:, 27] = pd[3]
        self.obs[:, 28] = pd[4]
        self.obs[:, 29] = pd[5]
        self.obs[:, 30] = pd[6]
        self.obs[:, 31] = vd[0]
        self.obs[:, 32] = vd[1]
        self.obs[:, 33] = vd[2]
        return self.obs

    def calculate_metrics(self) -> None:
        j_vel = torch.zeros(7)
        j_pos = torch.zeros(7)
        for i in range(7):
            ind = 2 * i
            j_pos[i] = self.obs[:, ind]
            j_vel[i] = self.obs[:, ind+1]


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
        # get trajectories and errors
        pd = torch.zeros(7)
        vd = torch.zeros(3)
        pd[0] = self.obs[:, 24]
        pd[1] = self.obs[:, 25]
        pd[2] = self.obs[:, 26]
        pd[3] = self.obs[:, 27]
        pd[4] = self.obs[:, 28]
        pd[5] = self.obs[:, 29]
        pd[6] = self.obs[:, 30]
        vd[0] = self.obs[:, 31]
        vd[1] = self.obs[:, 32]
        vd[2] = self.obs[:, 33]
        pe = torch.sum(torch.abs(hand_pos- pd))
        ve = torch.sum(torch.abs(hand_vel - vd))

        limits = self._sawyer.get_dof_limits()[:, self._joint_indices, :]
        frac = torch.zeros(7)
        for i in range(7):
            upper = limits[:, i, 1]
            lower = limits[:, i, 0]

            center = (lower + upper)/ 2

            frac[i] = torch.abs(j_pos[i] - center) / (upper - lower)


        # compute reward based on gripper pose and position vs trajectory, use get_current_time() and self.start_time
        klog = lambda x, l: 2 / (torch.exp(x * l) + torch.exp(-x * l))

        forces = self._sawyer.get_applied_joint_efforts()[:, self._joint_indices]
        for i in range(7):
            forces[:, i] = forces[:, i] / self._max_torque[0][i]


        reward = self.w * klog(pe, self.l1) + (1 - self.w) * klog(ve, self.l2) - 1 - (1/7) * torch.sum(torch.abs(forces)) - (1/14) * torch.sum(torch.abs(frac))


        #reward = klog() - 0.002 * torch.norm(self._sawyer.get_applied_joint_efforts())
        #reward = torch.where(reset, torch.ones_like(reward) * -2.0, reward)
        #reward = torch.where(hand_pos[6] < 0, torch.ones_like(reward) * -1.0, reward)
        #reward = torch.where(torch.max(torch.abs(j_vel)) > self._reset_vel, torch.ones_like(reward) * -1.0, reward)
        # compute penalty if jacobian is near singular, use get_jacobians
        return reward.item()

    def is_done(self) -> None:
        j_vel = torch.zeros(7)
        j_pos = torch.zeros(7)
        for i in range(7):
            ind = 2 * i
            j_pos[i] = self.obs[:, ind]
            j_vel[i] = self.obs[:, ind+1]

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

        pd = torch.zeros(7)
        vd = torch.zeros(3)
        pd[0] = self.obs[:, 24]
        pd[1] = self.obs[:, 25]
        pd[2] = self.obs[:, 26]
        pd[3] = self.obs[:, 27]
        pd[4] = self.obs[:, 28]
        pd[5] = self.obs[:, 29]
        pd[6] = self.obs[:, 30]
        vd[0] = self.obs[:, 31]
        vd[1] = self.obs[:, 32]
        vd[2] = self.obs[:, 33]

        pe = torch.sum(torch.abs(hand_pos- pd))
        ve = torch.sum(torch.abs(hand_vel - vd))

        limits = self._sawyer.get_dof_limits()
        reset = torch.tensor([0],dtype=torch.bool)
        for i in range(7):
            jo = int(self._joint_indices[i].item())
            if (limits[:, jo, 1] < j_pos[i]) or (limits[:, jo, 0] > j_pos[i]):
                reset = torch.tensor([1],dtype=torch.bool)
                break
        
        time = torch.tensor(self.simulation.current_time - self.start_time)
        # reset if sawyer joint velocities too high, dof limits exceeded, or trajectory finished
        # resets = torch.where(torch.max(torch.abs(j_vel)) > self._reset_vel, 1, 0)
        '''
        print("oof")
        print(hand_pos)
        print(pd)
        print(hand_vel)
        print(vd)

        print("error")
        print(pe)
        print(ve)
        '''
        resets = torch.where(time > 5, 1, 0)
        resets = torch.where(hand_pos[2] < 0, 1, resets)
        #resets = torch.where(pe > 5, 1, resets)
        #resets = torch.where(ve > 5, 1, resets)
        

        self.resets = torch.tensor([[resets.item()]])
        return resets.item()
