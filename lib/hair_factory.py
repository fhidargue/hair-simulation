import numpy as np
from ncca.ngl.vec3 import Vec3

from lib.constants import EPSILON, SURFACE_OFFSET
from lib.utils import (
    apply_face_transform,
    load_grayscale_mask,
    make_strand_test,
    rotate_direction,
)


def sample_mask(mask: np.ndarray, uv: np.ndarray) -> float:
    """
    mask: Height and Width in an array from 0 to 1 [0,1]
    uv: (2,) in [0,1]
    """
    height, width = mask.shape
    u = float(uv[0]) % 1.0
    v = float(uv[1]) % 1.0

    x = int(u * (width - 1))
    y = int((1.0 - v) * (height - 1))

    return float(mask[y, x])


def create_head_hair_from_mesh(
    hair_system,
    face_mesh,
    face_position,
    face_scale,
    face_rotation,
    strands,
    spacing,
    particles,
    make_strand_func,
    scalp_y_threshold=0.6,
    normal_y_threshold=0.3,
    is_testing=False,
    mask_path=None,
    mask_threshold=0.5,
):
    """
    Create hair strands on a head mesh by sampling scalp vertices.

    Args:
        hair_system (HairSystem): The hair system to which strands will be added.
        face_mesh (FaceMesh): The mesh of the face/head.
        face_position (Vec3): Position of the face in world coordinates.
        face_scale (Vec3): Scale of the face.
        face_rotation (Vec3): Rotation of the face.
        strands (int): Number of hair strands to create.
        spacing (float): Spacing between particles in a strand.
        particles (int): Number of particles per strand.
        make_strand_func (function): Function to create a hair strand.
        scalp_y_threshold (float): Threshold to identify scalp vertices based on Y position.
        normal_y_threshold (float): Minimum Y component of normal to consider a vertex for hair.
        is_testing (bool): If True, use a test strand creation method.
    """
    verts = np.asarray(face_mesh.vertices, dtype=np.float32)
    norms = np.asarray(face_mesh.normals, dtype=np.float32)

    if verts.size == 0 or norms.size == 0:
        raise RuntimeError("Face mesh has no valid vertices or normals")

    mask = None
    if mask_path is not None:
        mask = load_grayscale_mask(mask_path)

    if mask is not None and face_mesh.uvs is not None:
        uvs = np.asarray(face_mesh.uvs, dtype=np.float32)

        allowed = []
        weights = []

        for index in range(verts.shape[0]):
            m = sample_mask(mask, uvs[index])
            if m >= mask_threshold:
                allowed.append(index)
                weights.append(m)  # Color density as weight

        allowed = np.asarray(allowed, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32)

        if allowed.size == 0:
            raise RuntimeError(
                f"No vertices passed the mask threshold={mask_threshold}."
            )

        weights /= weights.sum() + EPSILON

        pick_n = min(strands, allowed.size)
        candidate_indices = np.random.choice(
            allowed, size=pick_n, replace=False, p=weights
        )

    else:
        values_y = verts[:, 1]
        min_y, max_y = values_y.min(), values_y.max()
        cut_y = min_y + (max_y - min_y) * scalp_y_threshold

        candidate_indices = np.where(values_y >= cut_y)[0]
        np.random.shuffle(candidate_indices)
        candidate_indices = candidate_indices[:strands]

    for index in candidate_indices:
        normal_local = norms[index].astype(np.float32)

        # If using a mask, trust the placement
        if mask is None:
            if normal_local[1] < normal_y_threshold:
                continue

        normal_local = normal_local / (np.linalg.norm(normal_local) + EPSILON)
        root_local = verts[index] * float(face_scale) + normal_local * (
            SURFACE_OFFSET * 3.0
        )
        root_world = apply_face_transform(root_local, face_position, 1.0, face_rotation)

        dir_world = rotate_direction(normal_local, face_rotation)
        dir_world = dir_world / (np.linalg.norm(dir_world) + EPSILON)

        if is_testing:
            strand = make_strand_test()
        else:
            strand = make_strand_func(
                spacing=spacing,
                length=particles * spacing,
                root_position=Vec3(
                    float(root_world[0]), float(root_world[1]), float(root_world[2])
                ),
                num_particles=particles,
                direction=Vec3(
                    float(dir_world[0]), float(dir_world[1]), float(dir_world[2])
                ),
                is_curly=(hair_system.hair_style == 1),  # Index for the Curly UI option
            )

        strand.root_local = root_local.copy()

        hair_system.add_strand(strand)
