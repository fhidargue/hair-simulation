import math

import numpy as np
from ncca.ngl.vec3 import Vec3
from PIL import Image

from lib.constants import HAIR_RADIUS
from module.hair_particle import HairParticle
from module.hair_strand import HairStrand


def make_strand_test():
    """
    Testing function for creating a dummy strand

    Returns:
        HairStrand: The created hair strand.
    """
    p1 = HairParticle(
        position_init=Vec3(0.0, 0.0, 0.0),
        _prev_position_init=Vec3(0.0, 0.0, 0.0),
        pinned=True,
        pin_position=Vec3(0.0, 0.0, 0.0),
    )

    p2 = HairParticle(
        position_init=Vec3(0.0, -1.0, 0.0),
        _prev_position_init=Vec3(0.0, -1.0, 0.0),
    )

    p3 = HairParticle(
        position_init=Vec3(0.0, -2.0, 0.0),
        _prev_position_init=Vec3(0.0, -2.0, 0.0),
    )

    return HairStrand(
        particles=[p1, p2, p3],
        _segment_length=1.0,
        root_position=Vec3(0.0, 0.0, 0.0),
        _stiffness=1.0,
    )


def make_strand(
    spacing,
    length,
    root_position,
    num_particles,
    direction,
    radius=HAIR_RADIUS,
    is_curly=False,
):
    """
    Creates a HairStrand growing along a given direction (mesh normal).

    Args:
        spacing (float): Spacing between particles in the strand.
        length (float): Total length of the strand.
        root_position (Vec3): The root position of the strand.
        num_particles (int): Number of particles in the strand.
        direction (Vec3): The growth direction of the strand.
    """
    direction = direction.normalize()
    particles = []

    for index in range(num_particles):
        offset = direction * (-index * spacing)

        pos = Vec3(
            root_position.x + offset.x,
            root_position.y + offset.y,
            root_position.z + offset.z,
        )

        particles.append(
            HairParticle(
                position_init=pos,
                _prev_position_init=pos,
                pinned=(index == 0),
                pin_position=pos if index == 0 else None,
            )
        )

    return HairStrand(
        particles=particles,
        root_position=root_position,
        _segment_length=spacing,
        _stiffness=1.0,
        radius=radius,
        is_curly=is_curly,
    )


def _euler_xyz_matrix_deg(rotation):
    rx = math.radians(rotation.x)
    ry = math.radians(rotation.y)
    rz = math.radians(rotation.z)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rx
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)

    # Ry
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)

    # Rz
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    # XYZ order
    return (Rz @ Ry @ Rx).astype(np.float32)


def apply_face_transform(vector, position, scale, rotation):
    """
    Applies scale, Euler rotation and translation to a local space vector.

    Args:
        vector (Vec3): 3D vector in local face space.
        position (Vec3): World space position of the face (translation).
        scale (float): Uniform scale applied before rotation.
        rotation (Vec3): Euler rotation in degrees (XYZ order).
    """
    v = np.asarray(vector, dtype=np.float32) * float(scale)
    R = _euler_xyz_matrix_deg(rotation)
    v = R @ v
    return np.array(
        [v[0] + position.x, v[1] + position.y, v[2] + position.z], dtype=np.float32
    )


def rotate_direction(direction, rotation):
    """
    Rotates a direction vector by the given Euler rotation without applying translation.

    Args:
        direction (array): 3D direction vector to rotate.
        rotation (Vec3): Euler rotation in degrees (XYZ order).
    """
    d = np.asarray(direction, dtype=np.float32)
    R = _euler_xyz_matrix_deg(rotation)
    return (R @ d).astype(np.float32)


def load_grayscale_mask(path: str) -> np.ndarray:
    """
    Loads a grayscale image and returns values in [0,1].
    White = 1 and Black = 0

    Args:
        path(str): Path to the grayscale UV mask.
    """
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def ngl_mat4_to_np(M) -> np.ndarray:
    """
    Convert ncca ngl Mat4 to numpy (4,4)
    """
    out = np.zeros((4, 4), dtype=np.float32)

    for c in range(4):
        for r in range(4):
            out[r, c] = float(M[c][r])
    return out
