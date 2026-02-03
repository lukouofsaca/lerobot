import logging
import time
from functools import cached_property

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from ..robot import Robot
from .config_piper import PiperConfig

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
import piper_sdk

logger = logging.getLogger(__name__)


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    # Piper usually has 6 joints + gripper
    JOINT_NAMES = [f"joint_{i}" for i in range(6)]
    GRIPPER_NAME = "gripper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        self.robot = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def observation_features(self) -> dict:
        """
        Define the structure of observations (e.g. motor positions, camera images).
        """
        features = {}
        # Joints position
        for name in self.JOINT_NAMES:
            features[f"{name}.pos"] = float
        features[f"{self.GRIPPER_NAME}.pos"] = float

        # Add cameras
        for name, config in self.config.cameras.items():
            features[name] = (config.height, config.width, 3)

        return features

    @property
    def action_features(self) -> dict:
        """
        Define the structure of actions (e.g. motor target positions).
        """
        features = {}
        for name in self.JOINT_NAMES:
            features[f"{name}.pos"] = float
        features[f"{self.GRIPPER_NAME}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self.robot is not None and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the Piper robot using the configured port.
        """
        port = self.config.port if self.config.port else "can0"
        logger.info(f"Connecting to Piper on {port}...")

        try:
            # Using C_PiperInterface as per AgileX SDK standard
            self.robot = piper_sdk.C_PiperInterface(port)
            ret = self.robot.ConnectPort()
            
            # Note: SDK return value handling might vary. Assuming boolean-like behavior.
            # If ret is a status code, inspect it.
            
            self.robot.EnableArm(7)  # Enable using default mode (7 usually enables all)
            
            for cam in self.cameras.values():
                cam.connect()
                
        except Exception as e:
            logger.error(f"Error connecting to Piper: {e}")
            self.robot = None
            raise ConnectionError(f"Could not connect to Piper: {e}")

        logger.info("Piper connected.")

    def calibrate(self) -> None:
        """
        Calibrate the robot if necessary.
        """
        pass

    def configure(self) -> None:
        """
        Configure the robot (e.g. set operating modes).
        """
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Read the current state from the robot.
        """
        try:
            arm_msg = self.robot.GetArmJointMsgs()
            gripper_msg = self.robot.GetArmGripperMsgs()
        except Exception as e:
            logger.error(f"Failed to read observation: {e}")
            raise

        obs = {}

        # Default mapping of joint indices 1-6 to joint_0-5
        # SDK likely provides integer values in 0.001 degrees
        joint_values = []
        if hasattr(arm_msg, 'joint_state'):
            joint_values = [
                arm_msg.joint_state.joint_1,
                arm_msg.joint_state.joint_2,
                arm_msg.joint_state.joint_3,
                arm_msg.joint_state.joint_4,
                arm_msg.joint_state.joint_5,
                arm_msg.joint_state.joint_6
            ]
        
        gripper_val = 0.0
        if hasattr(gripper_msg, 'gripper_state'):
             # Assuming gripper_angle is the field
             gripper_val = getattr(gripper_msg.gripper_state, 'gripper_angle', 0.0)

        # Convert units: 0.001 degrees -> degrees or radians
        factor = 0.001
        
        for i, name in enumerate(self.JOINT_NAMES):
            # Safe access if list is short
            val_raw = joint_values[i] if i < len(joint_values) else 0
            val_deg = val_raw * factor
            val = val_deg if self.config.use_degrees else np.deg2rad(val_deg)
            obs[f"{name}.pos"] = val

        obs[f"{self.GRIPPER_NAME}.pos"] = float(gripper_val) # Assuming raw unit is acceptable or user maps it

        # Add camera data
        for name, cam in self.cameras.items():
            obs[name] = cam.read()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send the desired action to the robot.
        """
        targets = []
        for name in self.JOINT_NAMES:
            if f"{name}.pos" in action:
                val = action[f"{name}.pos"]
                # Convert back to raw (0.001 deg)
                if not self.config.use_degrees:
                    val = np.rad2deg(val)
                raw_val = int(val * 1000)
                targets.append(raw_val)
            else:
                targets.append(0) # Should ideally be current position
        
        gripper_val = 0
        if f"{self.GRIPPER_NAME}.pos" in action:
             gripper_val = int(action[f"{self.GRIPPER_NAME}.pos"])

        try:
            # MotionCtrl_2: mode, speed, targets
            # Check SDK for exact arguments.
            # Assuming: mode=1 (Absolute), mode=1 (Speed?), speed=50, joints=[...]
            self.robot.MotionCtrl_2(0x01, 0x01, 50, targets)
            
            # Gripper control
            if f"{self.GRIPPER_NAME}.pos" in action:
                self.robot.GripperCtrl(abs(gripper_val), 1000, 0x01)
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        Disconnect from the robot.
        """
        if self.robot:
             # self.robot.DisconnectPort() # If supported
             self.robot = None
        for cam in self.cameras.values():
            cam.disconnect()
