import math
import numpy as np
from scipy import linalg
from toolbox import vector as vector_utils
from toolbox import graphics as graphics_utils


class BaseCamera:
    def __init__(self, size, near, far, clear_color=(1.0, 1.0, 1.0, 1.0)):
        self.size = size
        self.near = near
        self.far = far
        self.clear_color = clear_color
        if len(self.clear_color) == 3:
            self.clear_color = (*self.clear_color, 1.0)
        self.position = None
        self.up = None
        self.lookat = None

    @property
    def left(self):
        return -self.size[0] / 2

    @property
    def right(self):
        return self.size[0] / 2

    @property
    def top(self):
        return self.size[1] / 2

    @property
    def bottom(self):
        return -self.size[1] / 2

    @property
    def forward(self):
        return vector_utils.normalized(np.subtract(self.lookat, self.position))

    def projection_mat(self):
        raise NotImplementedError

    def rotation_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = vector_utils.normalized(
            np.cross(self.forward, self.up))
        rotation_mat[2, :] = -self.forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])
        return rotation_mat

    def translation_vec(self):
        rotation_mat = self.rotation_mat()
        return -rotation_mat.T @ self.position

    def view_mat(self):
        rotation_mat = self.rotation_mat()
        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

        return view_mat

    def cam_to_world(self):
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = self.rotation_mat().T
        cam_to_world[:3, 3] = self.position
        return cam_to_world

    def handle_mouse(self, last_pos, cur_pos):
        pass

    def apply_projection(self, points):
        homo = graphics_utils.euclidean_to_homogeneous(points)
        proj = self.projection_mat().dot(self.view_mat().dot(homo.T)).T
        proj = graphics_utils.homogeneous_to_euclidean(proj)[:, :2]
        proj = (proj + 1) / 2
        proj[:, 0] = (proj[:, 0] * self.size[0])
        proj[:, 1] = self.size[1] - (proj[:, 1] * self.size[1])
        return np.fliplr(proj)

    def get_position(self):
        return linalg.inv(self.view_mat())[:3, 3]

    def serialize(self):
        raise NotImplementedError()


class CalibratedCamera(BaseCamera):
    def __init__(self, extrinsic: np.ndarray, intrinsic: np.ndarray,
                 size, near, far, *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic

    def projection_mat(self):
        return graphics_utils.intrinsic_to_opengl_projection(
            self.intrinsic,
            self.left, self.right, self.top, self.bottom,
            self.near, self.far)

    def view_mat(self):
        return graphics_utils.extrinsic_to_opengl_modelview(self.extrinsic)

    def serialize(self):
        return {
            'type': 'calibrated',
            'size': self.size,
            'near': float(self.near),
            'far': float(self.far),
            'extrinsic': self.extrinsic.tolist(),
            'intrinsic': self.intrinsic.tolist(),
            'clear_color': self.clear_color,
        }


class PerspectiveCamera(BaseCamera):

    def __init__(self, size, near, far, fov, position, lookat, up,
                 *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)

        self.fov = fov
        self._position = np.array(position, dtype=np.float32)
        self.lookat = np.array(lookat, dtype=np.float32)
        self.up = vector_utils.normalized(np.array(up))

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position)

    def projection_mat(self):
        mat = util.transforms.perspective(
            self.fov, self.size[0] / self.size[1], self.near, self.far).T
        return mat

    def view_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = vector_utils.normalized(
            np.cross(self.forward, self.up))
        rotation_mat[2, :] = -self.forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

        return view_mat

    def serialize(self):
        return {
            'type': 'perspective',
            'size': self.size,
            'near': float(self.near),
            'far': float(self.far),
            'fov': self.fov,
            'position': self.position.tolist(),
            'lookat': self.lookat.tolist(),
            'up': self.up.tolist(),
            'clear_color': self.clear_color,
        }


class OrthographicCamera(BaseCamera):
    def __init__(self, size, near, far, position, lookat, up, *args, **kwargs):
        super().__init__(size, near, far, *args, **kwargs)
        self.lookat = lookat
        self.position = position
        self.up = up

    def projection_mat(self):
        return graphics_utils.ortho(
            self.left, self.right, self.bottom, self.top, self.near, self.far).T

    def serialize(self):
        return {
            'type': 'orthographic',
            'size': self.size,
            'near': float(self.near),
            'far': float(self.far),
            'fov': self.fov,
            'position': self.position.tolist(),
            'lookat': self.lookat.tolist(),
            'up': self.up.tolist(),
            'clear_color': self.clear_color,
        }


def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * math.cos(azimuth) * math.sin(elevation)
    y = radius * math.cos(elevation)
    z = radius * math.sin(azimuth) * math.sin(elevation)
    return x, y, z


def spherical_coord_to_cam(fov, azimuth, elevation, max_len=500, cam_dist=200):
    shape = (max_len * 2, max_len * 2)
    camera = PerspectiveCamera(
        size=shape, fov=fov, near=0.1, far=5000.0,
        position=(0, 0, -cam_dist), clear_color=(1, 1, 1, 1),
        lookat=(0, 0, 0), up=(0, 1, 0))
    camera.position = spherical_to_cartesian(cam_dist, azimuth, elevation)
    return camera
