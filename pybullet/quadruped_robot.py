import pybullet as p
import numpy as np
import re


class QuadrupedRobot(object):
    def __init__(self, pybullet_client, init_position, init_orientation, init_joints,
                 joint_stiffness=np.array([30.0, 30.0, 30.0] * 4), joint_damping=np.array([0.5, 0.5, 0.5] * 4)):
        self.pybullet_client = pybullet_client

        self.init_position = init_position
        self.init_orientation = init_orientation
        _, self.init_orientation_inv = p.invertTransform(position=[0, 0, 0], orientation=init_orientation)
        self.init_joints = init_joints
        self.joint_directions = np.array([1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1])
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping

        self.naming_pattern = {'base_link': re.compile(r'\w*floating_base\w*'),
                               'hip_joint': re.compile(r'\w+_hip_joint\w*'),
                               'calf_joint': re.compile(r'\w+_calf_joint\w*'),
                               'thigh_joint': re.compile(r'\w+_thigh_joint\w*'),
                               'foot_link': re.compile(r'\w+_foot_\w+')}

    def load_urdf(self, urdf_file):
        self.uid = p.loadURDF(urdf_file, useFixedBase=False, basePosition=self.init_position)
        self.load_joint_ids()

    def load_joint_ids(self):
        num_joints = self.pybullet_client.getNumJoints(self.uid)

        self.joint_name_to_id = {}
        self.joint_ids = []  # all motor joints
        self.hip_joint_ids = []  # hip joint indices only
        self.thigh_joint_ids = []  # thigh joint indices only
        self.calf_joint_ids = []  # calf joint indices only
        self.foot_link_ids = []  # foot joint indices
        self.body_link_id = [-1]  # just base link

        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.uid, i)
            joint_name, joint_id = joint_info[1].decode('UTF-8'), joint_info[0]
            self.joint_name_to_id[joint_name] = joint_id
            if self.naming_pattern['base_link'].match(joint_name):
                self._chassis_link_ids = [joint_id]
            elif self.naming_pattern['hip_joint'].match(joint_name):
                self.hip_joint_ids.append(joint_id)
            elif self.naming_pattern['thigh_joint'].match(joint_name):
                self.thigh_joint_ids.append(joint_id)
            elif self.naming_pattern['calf_joint'].match(joint_name):
                self.calf_joint_ids.append(joint_id)
            elif self.naming_pattern['foot_link'].match(joint_name):
                self.foot_link_ids.append(joint_id)
            else:
                continue
                # raise ValueError('Unknown category of joint %s' % joint_name)

        # everything associated with the leg links
        self.joint_ids.extend(self.hip_joint_ids)
        self.joint_ids.extend(self.thigh_joint_ids)
        self.joint_ids.extend(self.calf_joint_ids)
        self.joint_ids.sort()
        self.hip_joint_ids.sort()
        self.thigh_joint_ids.sort()
        self.calf_joint_ids.sort()
        self.foot_link_ids.sort()

    def load_init_pose(self):
        for i in range(len(self.joint_ids)):
            self.pybullet_client.resetJointState(self.uid,
                                                 self.joint_ids[i],
                                                 targetValue=self.init_joints[i],
                                                 targetVelocity=0)

    def get_observation(self):
        # TODO: reformat merging function, not divide functions to save processing time
        observation = []
        observation.extend(self.get_joint_position().tolist())
        observation.extend(self.get_joint_velocities().tolist())
        observation.extend(self.get_joint_torqes().tolist())
        observation.extend(list(self.get_body_orientation()))
        observation.extend(self.get_body_linear_velocity())
        observation.extend(self.get_body_angular_velocity())
        observation.extend(self.get_body_height())
        observation.extend(self.get_foot_contact())
        return observation

    def get_joint_position(self):
        joint_angles = [self.pybullet_client.getJointState(self.uid, joint_id)[0] for joint_id in self.joint_ids]
        joint_angles = np.multiply(joint_angles, self.joint_directions)
        return joint_angles

    def get_joint_velocities(self):
        joint_velocities = [self.pybullet_client.getJointState(self.uid, joint_id)[1] for joint_id in self.joint_ids]
        joint_velocities = np.multiply(joint_velocities, self.joint_directions)
        return joint_velocities

    def get_joint_torqes(self):
        joint_torques = [self.pybullet_client.getJointState(self.uid, joint_id)[3] for joint_id in self.joint_ids]
        joint_torques = np.multiply(joint_torques, self.joint_directions)
        return joint_torques

    def get_body_orientation(self):
        _, body_orientation = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        _, body_orientation = self.pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=body_orientation,
            positionB=[0, 0, 0],
            orientationB=self.init_orientation_inv)
        return body_orientation

    def get_body_linear_velocity(self):
        body_linear_velocity, _ = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_linear_velocity)

    def get_body_angular_velocity(self):
        _, body_angular_velocity = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_angular_velocity)

    def get_body_height(self):
        body_position, _ = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return [body_position[2]]

    def get_foot_contact(self):
        return [1, 2, 3, 4]
        # TODO

    def apply_torques(self, command):
        if len(command) != len(self.joint_ids):
            raise ValueError('Invalid shape of torque command and joints')
        for i in range(len(self.joint_ids)):
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.uid,
                                                       jointIndex=self.joint_ids[i],
                                                       controlMode=self.pybullet_client.TORQUE_CONTROL,
                                                       force=command[i])
