from dataclasses import dataclass, field
from typing import List

import numpy as np
from ncca.ngl.vec3 import Vec3

from lib.constants import (
    COLLISION_FRICTION,
    COLLISION_OFFSET,
    EPSILON,
    GRAVITY,
    MOUTH_COLLIDER_CENTER_OFFSET,
    MOUTH_COLLIDER_SCALE,
    NOSE_COLLIDER_CENTER_OFFSET,
    NOSE_COLLIDER_SCALE,
)
from lib.utils import _euler_xyz_matrix_deg, apply_face_transform
from module.hair_strand import HairStrand


@dataclass
class HairSystem:
    strands: List[HairStrand] = field(default_factory=list)
    gravity: Vec3 = field(default_factory=lambda: Vec3(0.0, -9.81, 0.0))
    wind: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))

    # Head variables
    head_position: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    head_rotation: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    head_radius: float = 2.0

    # Wind variables
    wind_enabled = False
    wind_strength = 20.0
    wind_radius = 20.0
    _time: float = 0.0

    # Hair variables
    hair_style: int = 0

    # Collision variables
    collision_enabled: bool = True
    collision_debug: bool = False

    # Base head collider
    collider_center_local_np: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    collider_radii_local_np: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )

    # Extra facial colliders
    nose_center_local_np: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    nose_radii_local_np: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32) * 0.25
    )

    mouth_center_local_np: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    mouth_radii_local_np: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32) * 0.35
    )

    # Head transform (turn world position into local)
    head_M_np: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    head_invM_np: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )

    def add_strand(self, strand: HairStrand):
        """
        Adds a hair strand to the hair system for simulation and rendering.

        Args:
            strand (HairStrand): The hair strand to add to the system.
        """
        self.strands.append(strand)

    @staticmethod
    def _head_matrix(position, rotation):
        """
        Computes the head transformation matrix.

        Args:
            position: Position of the head in world space.
            rotation: Rotation of the head in Euler angles (degrees).
        """
        R = _euler_xyz_matrix_deg(rotation)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = np.array([position.x, position.y, position.z], dtype=np.float32)
        return M

    @staticmethod
    def _apply_matrix_points(M, pts):
        """
        Applies a 4×4 transformation matrix to a set of 3D points.

        Args:
            M (np.ndarray): 4×4 transformation matrix.
            pts (np.ndarray): Array of 3D points with shape (N, 3).
        """
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        p4 = np.hstack([pts, ones])
        out = (p4 @ M.T)[:, :3]
        return out.astype(np.float32)

    def update(self, dt: float):
        """
        Updates the hair system simulation.

        Args:
            dt: Time step in seconds.
        """
        self._time += dt

        if self.wind_enabled:
            wind_dir = self._get_wind_direction()
        else:
            wind_dir = None

        for strand in self.strands:
            # Pin root to head (world space) using root_local
            if hasattr(strand, "root_local"):
                root_world = apply_face_transform(
                    strand.root_local,
                    self.head_position,
                    1.0,
                    self.head_rotation,
                )
                strand.positions_np[0] = root_world
                strand.prev_positions_np[0] = root_world

            strand.apply_global_forces(GRAVITY, Vec3(0.0, 0.0, 0.0))

            if self.wind_enabled and wind_dir is not None:
                positions = strand.positions_np
                num_particles = positions.shape[0]
                if num_particles <= 1:
                    continue

                indices = np.arange(num_particles, dtype=np.float32)
                tip_factor = indices / (num_particles - 1)
                tip_factor[0] = 0.0  # root stays fixed

                forces = wind_dir[None, :] * self.wind_strength * tip_factor[:, None]
                forces[:, 1] += 0.2 * tip_factor

                # Turbulence
                noise = np.stack(
                    [
                        np.sin(self._time * 3.0 + indices),
                        np.cos(self._time * 2.5 + indices * 1.3),
                        np.sin(self._time * 4.0 + indices * 0.7),
                    ],
                    axis=1,
                )
                noise_strength = 1.0 * tip_factor**1.5
                forces += noise * noise_strength[:, None]
                strand.acc_np[1:] += forces[1:]

            drag = -0.15 * (strand.positions_np - strand.prev_positions_np)
            strand.acc_np += drag

            strand.update(dt)

            if self.collision_enabled:
                self._collide_with_ellipsoid(
                    strand, self.nose_center_local_np, self.nose_radii_local_np
                )
                self._collide_with_ellipsoid(
                    strand, self.mouth_center_local_np, self.mouth_radii_local_np
                )
                self._collide_with_ellipsoid(
                    strand, self.collider_center_local_np, self.collider_radii_local_np
                )

    def _get_wind_direction(self):
        """
        Returns world-space wind direction blowing away from the face.
        """
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        rx = np.radians(self.head_rotation.x)
        ry = np.radians(self.head_rotation.y)
        rz = np.radians(self.head_rotation.z)

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = Rz @ Rx @ Ry
        wind_dir = -(R @ forward)
        return wind_dir / (np.linalg.norm(wind_dir) + EPSILON)

    def _compute_wind_at_point(self, point_np: np.ndarray) -> np.ndarray:
        """
        Simulates a fan in front of the face.

        Args:
            point_np: 3D point in world space as a numpy array.
        """
        if not self.wind_enabled:
            return np.zeros(3, dtype=np.float32)

        wind_dir = self._get_wind_direction()

        head_np = np.array(
            [self.head_position.x, self.head_position.y, self.head_position.z],
            dtype=np.float32,
        )

        to_point = point_np - head_np
        dist = np.linalg.norm(to_point)

        if dist > self.wind_radius:
            return np.zeros(3, dtype=np.float32)

        falloff = 1.0 - (dist / self.wind_radius)
        strength = self.wind_strength * falloff
        return wind_dir * strength

    def set_collider_from_face_mesh(
        self,
        face_mesh,
        shrink: float = 0.97,
        center_x_offset: float = 0.0,
        center_y_offset: float = 0.0,
        center_z_offset: float = 0.0,
        radius_x_scale: float = 1.0,
        radius_y_scale: float = 1.0,
        radius_z_scale: float = 1.0,
    ):
        """
        Build a local-space ellipsoid from the face mesh bounding box.

        Args:
            face_mesh: The face mesh object with a 'vertices' attribute.
            shrink: Factor to shrink the collider size.
            center_x_offset, center_y_offset, center_z_offset: Offsets to apply to the collider center.
            radius_x_scale, radius_y_scale, radius_z_scale: Scaling factors for the collider radii.
        """
        verts = np.asarray(face_mesh.vertices, dtype=np.float32)
        if verts.size == 0:
            raise RuntimeError("Face mesh has no vertices; cannot build collider.")

        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)

        center = 0.5 * (vmin + vmax)
        radii = 0.5 * (vmax - vmin)

        radii *= float(shrink)
        radii *= np.array(
            [radius_x_scale, radius_y_scale, radius_z_scale], dtype=np.float32
        )

        center = center + np.array(
            [center_x_offset, center_y_offset, center_z_offset], dtype=np.float32
        )
        radii = np.maximum(radii, np.array([0.05, 0.05, 0.05], dtype=np.float32))

        self.collider_center_local_np = center.astype(np.float32)
        self.collider_radii_local_np = radii.astype(np.float32)

        # Now build nose/mouth based on the updated main collider
        self.build_facial_feature_colliders(
            NOSE_COLLIDER_CENTER_OFFSET,
            NOSE_COLLIDER_SCALE,
            MOUTH_COLLIDER_CENTER_OFFSET,
            MOUTH_COLLIDER_SCALE,
        )

    def set_head_matrices(self, M_np: np.ndarray):
        """
        Call every frame with the SAME matrix used to draw the face.

        Args:
            M_np: 4x4 numpy array representing the head transformation matrix in world space.
        """
        self.head_M_np = M_np.astype(np.float32)
        self.head_invM_np = np.linalg.inv(self.head_M_np).astype(np.float32)

    def _world_to_head_local(self, position_world: np.ndarray) -> np.ndarray:
        """
        Transforms points from world space into head-local space.

        Args:
            position_world (np.ndarray): World space points with shape (N, 3).
        """
        N = position_world.shape[0]
        Pw = np.ones((N, 4), dtype=np.float32)
        Pw[:, :3] = position_world
        return (Pw @ self.head_invM_np.T)[:, :3]

    def _head_local_to_world(self, position_local: np.ndarray) -> np.ndarray:
        """
        Transforms points from head local space back into world space.

        Args:
            P_local (np.ndarray): Head local space points with shape (N, 3).
        """
        N = position_local.shape[0]
        Pl = np.ones((N, 4), dtype=np.float32)
        Pl[:, :3] = position_local
        return (Pl @ self.head_M_np.T)[:, :3]

    def _collide_with_ellipsoid(
        self, strand: HairStrand, center: np.ndarray, radii: np.ndarray
    ) -> int:
        """
        Resolves collisions between a hair strand and an ellipsoidal collider
        in head local space, pushing penetrated particles back to the surface.

        Args:
            strand (HairStrand): The hair strand to test and correct.
            center (np.ndarray): Ellipsoid center in head local space (3,).
            radii (np.ndarray): Ellipsoid radii along each axis (3,).
        """
        pos_w = strand.positions_np
        prev_w = strand.prev_positions_np

        if pos_w.shape[0] <= 1:
            return 0

        # Skip root (index 0)
        p_local = self._world_to_head_local(pos_w[1:])
        prev_local = self._world_to_head_local(prev_w[1:])

        q = p_local - center[None, :]
        inv_r = 1.0 / (radii[None, :] + EPSILON)
        s = q * inv_r
        d2 = (s * s).sum(axis=1)

        inside = d2 < 1.0
        if not np.any(inside):
            return 0

        si = s[inside]
        si_len = np.sqrt((si * si).sum(axis=1) + EPSILON)
        si_hat = si / si_len[:, None]

        surf_local = center[None, :] + si_hat * radii[None, :]

        n_local = si_hat * inv_r[0]
        n_local /= np.linalg.norm(n_local, axis=1)[:, None] + EPSILON

        corrected_local = surf_local + n_local * float(COLLISION_OFFSET)

        vel_local = p_local[inside] - prev_local[inside]
        vn_mag = (vel_local * n_local).sum(axis=1)
        vn = np.maximum(vn_mag, 0.0)[:, None] * n_local

        vt = vel_local - vn_mag[:, None] * n_local
        vt *= 1.0 - float(COLLISION_FRICTION)

        prev_local[inside] = corrected_local - (vn + vt)
        p_local[inside] = corrected_local

        # Write back into world
        pos_w[1:] = self._head_local_to_world(p_local)
        prev_w[1:] = self._head_local_to_world(prev_local)

        return int(np.count_nonzero(inside))

    def build_facial_feature_colliders(
        self,
        nose_center_offset,
        nose_radii_scale,
        mouth_center_offset,
        mouth_radii_scale,
    ):
        """
        Builds small nose & mouth ellipsoids in HEAD-LOCAL space.

        Args:
            nose_center_offset (array-like): Local-space offset from the head collider center to place the nose collider.
            nose_radii_scale (array-like): Per-axis scale (x,y,z) applied to head radii to size the nose collider.
            mouth_center_offset (array-like): Local-space offset from the head collider center to place the mouth collider.
            mouth_radii_scale (array-like): Per-axis scale (x,y,z) applied to head radii to size the mouth collider.
    """
        head_center = self.collider_center_local_np
        head_radii = self.collider_radii_local_np

        self.nose_center_local_np = head_center + np.array(
            nose_center_offset, dtype=np.float32
        )
        self.nose_radii_local_np = head_radii * np.array(
            nose_radii_scale, dtype=np.float32
        )

        self.mouth_center_local_np = head_center + np.array(
            mouth_center_offset, dtype=np.float32
        )
        self.mouth_radii_local_np = head_radii * np.array(
            mouth_radii_scale, dtype=np.float32
        )

        self.nose_radii_local_np = np.maximum(
            self.nose_radii_local_np, np.array([0.02, 0.02, 0.02], dtype=np.float32)
        )
        self.mouth_radii_local_np = np.maximum(
            self.mouth_radii_local_np, np.array([0.02, 0.02, 0.02], dtype=np.float32)
        )

    def build_facial_feature_colliders(
        self,
        nose_center_offset,
        nose_radii_scale,
        mouth_center_offset,
        mouth_radii_scale,
    ):
        """
        Builds head-local ellipsoidal colliders for the nose and mouth using offsets from the head collider and scaled radii.

        Args:
            nose_center_offset (array): (x, y, z) offset from head collider center to position the nose collider.
            nose_radii_scale (array): (sx, sy, sz) scale factors applied to head radii for the nose collider size.
            mouth_center_offset (array): (x, y, z) offset from head collider center to position the mouth collider.
            mouth_radii_scale (array): (sx, sy, sz) scale factors applied to head radii for the mouth collider size.
        """
        head_center = self.collider_center_local_np
        head_radii = self.collider_radii_local_np

        FRONT_SIGN = +1.0

        z_front = head_center[2] + FRONT_SIGN * head_radii[2]
        surface_push = 0.15

        # Nose
        nose_xy = np.array(
            [nose_center_offset[0], nose_center_offset[1]], dtype=np.float32
        )
        self.nose_center_local_np = head_center.copy()
        self.nose_center_local_np[0] += nose_xy[0]
        self.nose_center_local_np[1] += nose_xy[1]
        self.nose_center_local_np[2] = (
            z_front
            + FRONT_SIGN * head_radii[2] * surface_push
            + float(nose_center_offset[2])
        )

        self.nose_radii_local_np = head_radii * np.array(
            nose_radii_scale, dtype=np.float32
        )

        # Mouth
        mouth_xy = np.array(
            [mouth_center_offset[0], mouth_center_offset[1]], dtype=np.float32
        )
        self.mouth_center_local_np = head_center.copy()
        self.mouth_center_local_np[0] += mouth_xy[0]
        self.mouth_center_local_np[1] += mouth_xy[1]
        self.mouth_center_local_np[2] = (
            z_front
            + FRONT_SIGN * head_radii[2] * surface_push
            + float(mouth_center_offset[2])
        )

        self.mouth_radii_local_np = head_radii * np.array(
            mouth_radii_scale, dtype=np.float32
        )
