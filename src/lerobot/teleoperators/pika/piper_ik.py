import math
import os

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def matrix_to_xyzrpy(matrix: np.ndarray) -> list[float]:
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    transformation_matrix = np.eye(4)
    a = np.cos(yaw)
    b = np.sin(yaw)
    c = np.cos(pitch)
    d = np.sin(pitch)
    e = np.cos(roll)
    f = np.sin(roll)
    de = d * e
    df = d * f
    transformation_matrix[0, 0] = a * c
    transformation_matrix[0, 1] = a * df - b * e
    transformation_matrix[0, 2] = b * f + a * de
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = b * c
    transformation_matrix[1, 1] = a * e + b * df
    transformation_matrix[1, 2] = b * de - a * f
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -d
    transformation_matrix[2, 1] = c * f
    transformation_matrix[2, 2] = c * e
    transformation_matrix[2, 3] = z
    return transformation_matrix


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    rot = np.asarray(matrix, dtype=float)[:3, :3]
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot[2, 1] - rot[1, 2]) * s
        qy = (rot[0, 2] - rot[2, 0]) * s
        qz = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    return np.array([qx, qy, qz, qw], dtype=float)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw], dtype=float)


def euler_from_quaternion(quaternion_xyzw: list[float] | np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = quaternion_xyzw

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class OneEuroFilter:
    def __init__(self, t0: float, x0: float, dx0: float = 0.0, min_cutoff: float = 0.1, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class PiperFK:
    def __init__(self, urdf_path: str, package_dirs: str | list[str], gripper_xyzrpy: list[float], lift: bool = False):
        self.lift = lift
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        second_matrix = create_transformation_matrix(*gripper_xyzrpy)
        self.last_matrix = np.dot(first_matrix, second_matrix)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(
                    pin.Quaternion(q[3], q[0], q[1], q[2]),
                    np.array(
                        [
                            self.last_matrix[0, 3],
                            self.last_matrix[1, 3],
                            self.last_matrix[2, 3],
                        ]
                    ),
                ),
                pin.FrameType.OP_FRAME,
            )
        )

    def get_pose(self, q: list[float] | np.ndarray) -> list[float]:
        index = 6 + (1 if self.lift else 0)
        q_np = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.concatenate([q_np], axis=0))
        end_pose = create_transformation_matrix(
            self.reduced_robot.data.oMi[index].translation[0],
            self.reduced_robot.data.oMi[index].translation[1],
            self.reduced_robot.data.oMi[index].translation[2],
            math.atan2(self.reduced_robot.data.oMi[index].rotation[2, 1], self.reduced_robot.data.oMi[index].rotation[2, 2]),
            math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
            math.atan2(self.reduced_robot.data.oMi[index].rotation[1, 0], self.reduced_robot.data.oMi[index].rotation[0, 0]),
        )
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)


class PiperIK:
    def __init__(self, urdf_path: str, package_dirs: str | list[str], gripper_xyzrpy: list[float], lift: bool = False):
        self.lift = lift
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        second_matrix = create_transformation_matrix(*gripper_xyzrpy)
        self.last_matrix = np.dot(first_matrix, second_matrix)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(
                    pin.Quaternion(q[3], q[0], q[1], q[2]),
                    np.array(
                        [
                            self.last_matrix[0, 3],
                            self.last_matrix[1, 3],
                            self.last_matrix[2, 3],
                        ]
                    ),
                ),
                pin.FrameType.OP_FRAME,
            )
        )

        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model,
            urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=package_dirs,
        )
        for i in range(4, 11):
            for j in range(0, 4):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.c_tf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.c_tf],
            [
                casadi.vertcat(
                    cpin.log6(self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.c_tf)).vector,
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)

        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]
        ori_error = error_vec[3:]

        weight_position = 1.0
        weight_orientation = 0.1
        total_cost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        regularization = casadi.sumsqr(self.var_q)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(20 * total_cost + 0.01 * regularization)

        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": 50,
                "tol": 1e-4,
            },
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(
        self,
        target_pose_4x4: np.ndarray,
        gripper: float = 0,
        motorstate: list[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | str, bool]:
        gripper_q = np.array([gripper / 2.0, -gripper / 2.0], dtype=float)
        if motorstate is not None:
            self.init_data = np.asarray(motorstate, dtype=float)
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, target_pose_4x4)

        try:
            self.opti.solve_limited()
            sol_q = np.asarray(self.opti.value(self.var_q), dtype=float).reshape(-1)

            if self.init_data is not None:
                max_diff = float(np.max(np.abs(self.history_data - sol_q)))
                self.init_data = sol_q.copy()
                if max_diff > (30.0 / 180.0 * math.pi):
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q.copy()

            self.history_data = sol_q.copy()
            tau_ff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                np.zeros(self.reduced_robot.model.nv),
                np.zeros(self.reduced_robot.model.nv),
            )

            is_collision = self._check_self_collision(sol_q, gripper_q)
            return sol_q, tau_ff, not is_collision
        except Exception:
            return None, "", False

    def _check_self_collision(self, q: np.ndarray, gripper: np.ndarray) -> bool:
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        return pin.computeCollisions(self.geom_model, self.geometry_data, False)

    def get_pose(self, q: list[float] | np.ndarray) -> list[float]:
        index = 6 + (1 if self.lift else 0)
        q_np = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.concatenate([q_np], axis=0))
        end_pose = create_transformation_matrix(
            self.reduced_robot.data.oMi[index].translation[0],
            self.reduced_robot.data.oMi[index].translation[1],
            self.reduced_robot.data.oMi[index].translation[2],
            math.atan2(self.reduced_robot.data.oMi[index].rotation[2, 1], self.reduced_robot.data.oMi[index].rotation[2, 2]),
            math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
            math.atan2(self.reduced_robot.data.oMi[index].rotation[1, 0], self.reduced_robot.data.oMi[index].rotation[0, 0]),
        )
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)
