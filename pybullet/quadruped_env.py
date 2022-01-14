import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np

from quadruped_robot import QuadrupedRobot


class Config():
    def __init__(self):
        '''
            TODO: Move these config to yaml files
        '''
        self.is_render = True

        self.a1_urdf = './asset/urdfs/a1_description/urdf/a1.urdf'
        self.init_orientation = (0, 0, 0, 1)
        self.init_position = (0, 0, 0)

        self.num_observations = 3 * 12 + 3 * 3 + 1 + 4  # joint(pos + vel + torques) + body(orientation + linear vel + ang vel) + height + foot_contact
        self.max_episode_length = 20
        self.action_scale = 0.5
        self.control_mode = 'JointTorque'  # Modes: 'JointTorque','PDJoint', 'PDCartesian'
        if self.control_mode == 'JointTorque':
            self.num_actions = 12  # num of DOF
        if self.control_mode == 'PDJoint':
            self.num_actions = 12  # num of DOF
        elif self.control_mode == 'PDCartesian':
            self.num_actions = 12  # x,y,z * 4
        self.action_clip = 1.0
        self.observation_clip = 50.0
        self.joint_pd_kp = np.array([30.0, 30.0, 30.0] * 4)
        self.joint_pd_kd = np.array([0.5, 0.5, 0.5] * 4)

        self.start_pose = [0, 0, 0.38, 0, 0, 0, 1]
        self.init_joints = np.array([0.0, np.pi/4-0.15, -np.pi/2] * 4) # hip -> thigh -> calf, FR -> FL -> RR- > RL
        self.upper_joint_limit = np.array([0.802851455917, 4.18879020479, -0.916297857297] * 4)
        self.lower_joint_limit = np.array([-0.802851455917, -1.0471975512, -2.69653369433] * 4)
        self.torque_limits = np.array([33.5, 33.5, 33.5]*4)
        self.joint_stiffness = np.array([30.0, 30.0, 30.0] * 4)
        self.joint_damping = np.array([2.0, 2.0, 2.0] * 4)


class QuadrupedEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        # Load parameters configuration
        self.temp_config = Config()
        self.max_steps = 100
        self.run_steps = 0

        # Connect to Pybullet
        if self.temp_config.is_render:
            self.pybullet_client = bc.BulletClient(connection_mode=p.GUI)
            p.setTimeStep(1/50)
        else:
            self.pybullet_client = bc.BulletClient()
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set up environment specifications
        self.a1 = QuadrupedRobot(self.pybullet_client,
                                 self.temp_config.init_position,
                                 self.temp_config.init_orientation,
                                 joint_stiffness=self.temp_config.joint_stiffness,
                                 joint_damping=self.temp_config.joint_damping)
        self.action_space = spaces.Box(np.array([-self.temp_config.action_clip] * self.temp_config.num_actions),
                                       np.array([self.temp_config.action_clip] * self.temp_config.num_actions))
        self.observation_space = spaces.Box(np.array([-self.temp_config.observation_clip] * self.temp_config.num_observations),
                                            np.array([self.temp_config.observation_clip] * self.temp_config.num_observations))

        # Add viewer
        self.pybullet_client.resetDebugVisualizerCamera(cameraDistance=1.5,
                                                        cameraYaw=0,
                                                        cameraPitch=-40,
                                                        cameraTargetPosition=[0.55,-0.35,0.2]
                                                        )

    def reset(self):
        self.run_steps = 0

        # Reset the simulation
        self.pybullet_client.resetSimulation()
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # turn off redering befor re-loading
        self.pybullet_client.setGravity(0, 0, -10)

        # Load objects
        plane_uid = self.pybullet_client.loadURDF("plane.urdf", basePosition=[0,0,-0.65])

        self.a1.load_urdf(self.temp_config.a1_urdf)
        # TODO set initial state

        # Getting the observation & info
        observation = self.a1.get_observation()

        # Return
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) # turn on redering after loading
        return observation

    def step(self, action):
        # Setup
        self.run_steps += 1
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Step simulation
        self.pybullet_client.stepSimulation()
        self.apply_action(action)

        # Get the observation & info
        observation = self.a1.get_observation()
        info = 'None'

        # Get reward & done
        if self.run_steps > self.max_steps:
            reward = 0
            done = True
        else:
            reward = 0
            done = False
        
        # Return
        return observation, reward, done, info

    def apply_action(self, action):
        action = np.clip(action, -1,  1)
        if self.temp_config.control_mode == 'JointTorque':
            command = action * self.temp_config.torque_limits
            command *= self.temp_config.action_scale
            command = np.clip(command, -self.temp_config.torque_limits, self.temp_config.torque_limits)
            self.a1.apply_torques(command)
    
    def render(self, mode='human'):
        view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
                            cameraTargetPosition=[0.7,0,0.05],
                            distance=.7,
                            yaw=90,
                            pitch=-70,
                            roll=0,
                            upAxisIndex=2)
        proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(
                            fov=60,
                            aspect=float(960) /720,
                            nearVal=0.1,
                            farVal=100.0)
        (_, _, px, _, _) = self.pybullet_client.getCameraImage(
                            width=960,
                            height=720,
                            viewMatrix=view_matrix,
                            projectionMatrix=proj_matrix,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
        
    def close(self):
        self.pybullet_client.disconnect()
