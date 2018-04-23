import math

from rendkit.camera import PerspectiveCamera


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
