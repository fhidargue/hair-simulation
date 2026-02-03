import numpy as np
import OpenGL.GL as gl
from ncca.ngl import ShaderLib

from lib.constants import EPSILON, FOLLOWER_SPREAD, MAX_ROOT_RADIUS


class HairRibbonRenderer:
    """
    Expands hair strand centerlines into camera facing ribbon quads.
    """

    def __init__(self):
        self.ribbon_vao = None
        self.ribbon_vbo = None
        self.ribbon_segment_data = None
        self.ribbon_segment_count = 0
        self.thickness_scale = 0.0
        self.dirty = True

        # Follower settings
        self.followers_per_segment = 1
        self.follower_spread = FOLLOWER_SPREAD
        self.follower_seed = 1337
        self.max_root_radius = MAX_ROOT_RADIUS

        # GPU resources
        self.seg_tbo = None
        self.seg_tex = None
        self.seg_tex_unit = 0
        self.tbo_data = None

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True) + EPSILON
        return v / n

    def _build_frame(self, tangents: np.ndarray):
        """
        For each tangent, build two perpendicular directions (side, binormal)
        to offset followers around the guide.
        """
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # If tangent almost parallel to up, switch up axis (prevents near-zero cross)
        dot = np.abs((tangents * up[None, :]).sum(axis=1))
        up_alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        use_alt = dot > 0.9
        ups = np.where(use_alt[:, None], up_alt[None, :], up[None, :])

        side = np.cross(tangents, ups)
        side = self._normalize(side)
        bino = np.cross(tangents, side)
        bino = self._normalize(bino)
        return side, bino

    def _follower_offsets_for_strand(
        self, strand_index: int, positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute follower offsets for a given strand positions P (N x 3).

        Args:
            strand_index: Index of the strand (for random seed)
            P: Strand positions (N x 3)
        """
        K = int(max(1, self.followers_per_strand))
        N = positions.shape[0]
        if K == 1 or N < 2:
            return np.zeros((1, N, 3), dtype=np.float32)

        # Tangent per point (central difference)
        prev = positions[np.maximum(np.arange(N) - 1, 0)]
        next = positions[np.minimum(np.arange(N) + 1, N - 1)]
        tangents = self._normalize((next - prev).astype(np.float32))

        side, bino = self._build_frame(tangents)

        # Root to tip taper
        t = np.linspace(0.0, 1.0, N, dtype=np.float32)
        taper = (1.0 - t) ** float(self.follower_taper_power)

        offsets = np.zeros((K, N, 3), dtype=np.float32)

        # follower 0 has no offset
        offsets[0, :, :] = 0.0

        for k in range(1, K):
            seed = int(self.follower_seed + strand_index * 10007 + k * 97)
            rng = np.random.RandomState(seed)

            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radius = float(rng.uniform(0.35, 1.0)) * float(self.follower_spread)

            dir_vec = (np.cos(angle) * side) + (np.sin(angle) * bino)
            offsets[k, :, :] = dir_vec * (radius * taper)[:, None]

        return offsets

    def create_buffers(self, hair_system):
        """
        Creates the hair centerline VBO and EBO from the given HairSystem.

        Args:
            hair_system (HairSystem): The hair system containing strands to render.
        """
        # Count guide segments only
        total_segments = 0

        for strand in hair_system.strands:
            n = strand.positions_np.shape[0]
            if n >= 2:
                total_segments += n - 1

        self.ribbon_segment_count = int(total_segments)
        self.tbo_data = np.zeros((self.ribbon_segment_count * 2, 4), dtype=np.float32)
        write_seg = 0

        for strand in hair_system.strands:
            P = strand.positions_np.astype(np.float32)
            R = strand.radius_np.astype(np.float32)
            n = P.shape[0]
            if n < 2:
                continue

            A = P[:-1]
            B = P[1:]
            rA = R[:-1]
            rB = R[1:]

            s = A.shape[0]
            i0 = write_seg * 2

            # Join A and rA
            self.tbo_data[i0 : i0 + 2 * s : 2, 0:3] = A
            self.tbo_data[i0 : i0 + 2 * s : 2, 3] = rA

            # Join B and rB
            self.tbo_data[i0 + 1 : i0 + 2 * s : 2, 0:3] = B
            self.tbo_data[i0 + 1 : i0 + 2 * s : 2, 3] = rB
            write_seg += s

        if write_seg != self.ribbon_segment_count:
            self.ribbon_segment_count = write_seg
            self.tbo_data = self.tbo_data[: self.ribbon_segment_count * 2]

        # VAO
        if getattr(self, "ribbon_vao", None):
            try:
                gl.glDeleteVertexArrays(1, [self.ribbon_vao])
            except Exception:
                pass
        self.ribbon_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.ribbon_vao)
        gl.glBindVertexArray(0)

        # TBO plus texture
        if getattr(self, "seg_tex", None):
            try:
                gl.glDeleteTextures(1, [self.seg_tex])
            except Exception:
                pass
        if getattr(self, "seg_tbo", None):
            try:
                gl.glDeleteBuffers(1, [self.seg_tbo])
            except Exception:
                pass

        self.seg_tbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, self.seg_tbo)
        gl.glBufferData(
            gl.GL_TEXTURE_BUFFER,
            self.tbo_data.nbytes,
            self.tbo_data,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, 0)

        self.seg_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.seg_tex)
        gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_RGBA32F, self.seg_tbo)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)

        self.dirty = True

    def render(self, camera, color, camera_pos_np):
        """
        Renders hair ribbons on the GPU using instanced camera-facing quads expanded from hair segments.

        Args:
            camera: Active camera providing the view projection matrix.
            color: Hair color (Vec3) passed to the ribbon fragment shader.
            camera_pos_np (np.ndarray): World space camera position as a length 3 numpy array.
        """
        if self.ribbon_vao is None or self.seg_tex is None:
            return
        if self.thickness_scale <= 0.0:
            return
        if self.ribbon_segment_count <= 0:
            return

        K = int(max(1, self.followers_per_segment))
        instance_count = int(self.ribbon_segment_count) * K

        ShaderLib.set_uniform("MVP", camera.get_vp())
        ShaderLib.set_uniform("color", color.x, color.y, color.z)
        ShaderLib.set_uniform(
            "camera_position",
            float(camera_pos_np[0]),
            float(camera_pos_np[1]),
            float(camera_pos_np[2]),
        )
        ShaderLib.set_uniform("thickness", float(self.thickness_scale))
        ShaderLib.set_uniform("followers_per_segment", int(K))
        ShaderLib.set_uniform("follower_spread", float(self.follower_spread))
        ShaderLib.set_uniform("follower_seed", int(self.follower_seed))
        ShaderLib.set_uniform("max_root_radius", float(self.max_root_radius))

        gl.glActiveTexture(gl.GL_TEXTURE0 + self.seg_tex_unit)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self.seg_tex)
        ShaderLib.set_uniform("segmentTex", int(self.seg_tex_unit))

        gl.glBindVertexArray(self.ribbon_vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, instance_count)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)

    def update_segments_gpu(self, hair_system):
        """
        Updates the GPU segment buffer (TBO) with the latest simulated strand positions and radii.

        Args:
            hair_system: HairSystem containing the current strand particle positions and per particle radii.
        """
        if self.seg_tbo is None or self.tbo_data is None:
            return

        # Recompute expected segments (guides only)
        expected = 0
        for strand in hair_system.strands:
            n = strand.positions_np.shape[0]
            if n >= 2:
                expected += n - 1

        if (
            expected != int(self.ribbon_segment_count)
            or self.tbo_data.shape[0] != expected * 2
        ):
            self.create_buffers(hair_system)
            return

        write_seg = 0

        for strand in hair_system.strands:
            P = strand.positions_np.astype(np.float32)
            R = strand.radius_np.astype(np.float32)
            n = P.shape[0]
            if n < 2:
                continue

            A = P[:-1]
            B = P[1:]
            rA = R[:-1]
            rB = R[1:]
            s = A.shape[0]

            i0 = write_seg * 2
            self.tbo_data[i0 : i0 + 2 * s : 2, 0:3] = A
            self.tbo_data[i0 : i0 + 2 * s : 2, 3] = rA
            self.tbo_data[i0 + 1 : i0 + 2 * s : 2, 0:3] = B
            self.tbo_data[i0 + 1 : i0 + 2 * s : 2, 3] = rB

            write_seg += s

        # Upload to GPU
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, self.seg_tbo)
        gl.glBufferSubData(gl.GL_TEXTURE_BUFFER, 0, self.tbo_data.nbytes, self.tbo_data)
        gl.glBindBuffer(gl.GL_TEXTURE_BUFFER, 0)

    def build_cpu_tbo_data(self, hair_system):
        """
        Pure CPU packing for unit tests.
        """
        total_segments = 0
        for strand in hair_system.strands:
            n = strand.positions_np.shape[0]
            if n >= 2:
                total_segments += (n - 1)

        segment_count = int(total_segments)
        tbo_data = np.zeros((segment_count * 2, 4), dtype=np.float32)

        write_seg = 0
        for strand in hair_system.strands:
            P = strand.positions_np.astype(np.float32)
            R = strand.radius_np.astype(np.float32)
            n = P.shape[0]

            if n < 2:
                continue

            A = P[:-1]
            B = P[1:]
            rA = R[:-1]
            rB = R[1:]
            s = A.shape[0]

            i0 = write_seg * 2
            tbo_data[i0 : i0 + 2 * s : 2, 0:3] = A
            tbo_data[i0 : i0 + 2 * s : 2, 3] = rA
            tbo_data[i0 + 1 : i0 + 2 * s : 2, 0:3] = B
            tbo_data[i0 + 1 : i0 + 2 * s : 2, 3] = rB

            write_seg += s

        # safety
        if write_seg != segment_count:
            segment_count = write_seg
            tbo_data = tbo_data[: segment_count * 2]

        return segment_count, tbo_data
