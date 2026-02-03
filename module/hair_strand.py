from dataclasses import dataclass, field
from typing import List

import numpy as np
from ncca.ngl.vec3 import Vec3

from lib.constants import (
    CURL_AMPLITUDE,
    CURL_FREQUENCY,
    CURL_STRENGTH,
    EPSILON,
    HAIR_RADIUS,
)
from module.hair_particle import HairParticle


@dataclass
class HairStrand:
    particles: List[HairParticle]
    root_position: Vec3
    _segment_length: float
    _stiffness: float = 1.0

    positions_np: np.ndarray = field(init=False)
    prev_positions_np: np.ndarray = field(init=False)
    acc_np: np.ndarray = field(init=False)
    pinned_np: np.ndarray = field(init=False)

    # Radius variables
    radius: float = HAIR_RADIUS
    radius_np: np.ndarray = field(init=False)

    # Curl variables
    is_curly: bool = False
    curl_offset_np: np.ndarray = field(init=False)
    curl_amplitude: float = CURL_AMPLITUDE
    curl_frequency: float = CURL_FREQUENCY

    rest_lengths_np: np.ndarray = field(init=False)
    rest_positions_np: np.ndarray = field(init=False)

    # Precomputed constraint weights
    correction_weight_start_np: np.ndarray = field(init=False)
    correction_weight_end_np: np.ndarray = field(init=False)

    def __post_init__(self):
        num_particles = len(self.particles)

        self.positions_np = np.zeros((num_particles, 3), dtype=np.float32)
        self.prev_positions_np = np.zeros((num_particles, 3), dtype=np.float32)
        self.acc_np = np.zeros((num_particles, 3), dtype=np.float32)
        self.pinned_np = np.zeros((num_particles,), dtype=bool)
        self.radius_np = np.full((num_particles,), float(self.radius), dtype=np.float32)

        for index, particle in enumerate(self.particles):
            self.positions_np[index] = (
                particle.position_init.x,
                particle.position_init.y,
                particle.position_init.z,
            )
            self.prev_positions_np[index] = self.positions_np[index]
            self.pinned_np[index] = particle.pinned
            particle.attach(self, index)

        # Root to tip taper
        if num_particles > 1:
            t = np.linspace(0.0, 1.0, num_particles, dtype=np.float32)
            self.radius_np *= 1.0 - 0.7 * t  # ~30% from the root

        # Rest lengths based on initial positions
        delta = self.positions_np[1:] - self.positions_np[:-1]
        self.rest_lengths_np = np.sqrt((delta * delta).sum(axis=1) + 1e-12).astype(
            np.float32
        )

        # Curl offsets
        self.curl_offset_np = np.zeros_like(self.positions_np, dtype=np.float32)

        # Default rest pose is initial pose
        self.rest_positions_np = self.positions_np.copy()

        if self.is_curly and num_particles >= 2:
            strand_len = float(self._segment_length * (num_particles - 1))
            amplitude = min(float(self.curl_amplitude), 0.9 * strand_len)

            frequency = self.curl_frequency
            self.curl_phase = np.random.uniform(0.0, 2.0 * np.pi)

            # Strand direction
            strand_dir = self.positions_np[-1] - self.positions_np[0]
            strand_dir /= np.linalg.norm(strand_dir) + EPSILON

            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            if abs(np.dot(up, strand_dir)) > 0.9:
                up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            side = np.cross(strand_dir, up)
            side /= np.linalg.norm(side) + EPSILON

            # Build zigzag offsets
            # Inspired by Jon Macey, OpenGLPrimRestart/FasterVersion.py, Bournemouth University
            # Code referenced from: https://github.com/NCCA/PyNGLDemos/blob/main/OpenGLPrimRestart/FasterVersion.py
            # Date accessed: 23 December, 2025
            for index in range(1, num_particles):
                normalized_index = index / (num_particles - 1)
                radius = amplitude * normalized_index
                angle = (2.0 * np.pi * frequency * normalized_index) + self.curl_phase
                self.curl_offset_np[index] = side * (np.sin(angle) * radius)

            self.rest_positions_np = self.positions_np + self.curl_offset_np

            # Start in rest pose to avoid explosion
            self.positions_np[:] = self.rest_positions_np
            self.prev_positions_np[:] = self.rest_positions_np
            self.acc_np[:] = 0.0

        # Precompute persegment constraint weights
        segment_start_free = (~self.pinned_np[:-1]).astype(np.float32)
        segment_end_free = (~self.pinned_np[1:]).astype(np.float32)
        free_count = segment_start_free + segment_end_free
        inverse_free = 1.0 / (free_count + EPSILON)

        self.correction_weight_start_np = segment_start_free * inverse_free
        self.correction_weight_end_np = segment_end_free * inverse_free

    def update(self, dt: float, collide_fn=None):
        """
        Advances the hair strand simulation by one timestep using Verlet integration 
        and optional collision handling.

        Args:
            dt (float): Time step in seconds.
            collide_fn (callable): Optional collision callback applied after constraint solving.
        """
        ITERATIONS = 6 if self._segment_length * len(self.particles) > 0.3 else 4
        DAMPING = 0.96

        # Verlet integration with damping
        vel = (self.positions_np - self.prev_positions_np) * DAMPING
        self.prev_positions_np[:] = self.positions_np
        self.positions_np[:] = self.positions_np + vel + self.acc_np * (dt * dt)
        self.acc_np[:] = 0.0

        # Constraints collision inside the loop (most stable)
        for _ in range(ITERATIONS):
            self.enforce_constraints()

        self.apply_curl_bias(CURL_STRENGTH)

        # Collide outside the loop
        if collide_fn is not None:
            collide_fn(self)

    def enforce_constraints(self):
        """
        Enforces distance constraints between consecutive particles to maintain strand segment rest lengths,
        respecting pinned particles and strand stiffness.
        """
        if self._stiffness <= 0.0:
            return

        position = self.positions_np
        rest = self.rest_lengths_np

        # Segment vectors
        delta = position[1:] - position[:-1]

        # Segment lengths
        distance = np.sqrt((delta * delta).sum(axis=1) + 1e-12)

        # Constraint correction factor along delta
        diff = ((distance - rest) / distance) * self._stiffness
        corr = delta * diff[:, None]

        # Apply correction with pinned aware weights
        position[:-1] += corr * self.correction_weight_start_np[:, None]
        position[1:] -= corr * self.correction_weight_end_np[:, None]

    def apply_global_forces(self, gravity: Vec3, wind: Vec3):
        """
        Applies gravity and wind forces to all non-pinned particles in the strand.
        """
        gravity_force = np.array([gravity.x, gravity.y, gravity.z], dtype=np.float32)
        wind_force = np.array([wind.x, wind.y, wind.z], dtype=np.float32)

        mask = ~self.pinned_np
        self.acc_np[mask] += gravity_force + wind_force

    def apply_curl_bias(self, strength: float):
        """
        Adds a lateral zig-zag bias without killing velocity.

        Args:
            strength (float): The strength of the curly hair inside the mesh.
        """
        if not self.is_curly:
            return

        # Skip root
        self.positions_np[1:] += self.curl_offset_np[1:] * strength
