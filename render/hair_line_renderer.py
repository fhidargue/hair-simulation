import ctypes

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import ShaderLib, Transform


class HairLineRenderer:
    """
    Manages GPU buffers and rendering for hair centerlines (and optional points).
    Vertex format: vec4 (xyz + radius) at location 0.
    """

    def __init__(self):
        self.hair_vao = None
        self.hair_vbo = None
        self.hair_ebo = None
        self.hair_vertex_data = None
        self.hair_index_data = None
        self.hair_num_indices = 0
        self.restart_index = np.iinfo(np.uint32).max

    def create_buffers(self, hair_system):
        """
        Creates the hair centerline VBO and EBO from the given HairSystem.

        Args:
            hair_system (HairSystem): The hair system containing strands to render.
        """
        vertices = []
        indices = []
        index = 0

        for strand in hair_system.strands:
            n = strand.positions_np.shape[0]
            packed = np.hstack([strand.positions_np, strand.radius_np[:, None]]).astype(
                np.float32
            )
            vertices.extend(packed.tolist())

            indices.extend(range(index, index + n))
            index += n
            indices.append(self.restart_index)

        self.hair_vertex_data = np.asarray(vertices, dtype=np.float32)
        self.hair_index_data = np.asarray(indices, dtype=np.uint32)
        self.hair_num_indices = self.hair_index_data.size

        # Delete old buffers if rebuilding
        if self.hair_vao is not None:
            try:
                gl.glDeleteVertexArrays(1, [self.hair_vao])
                gl.glDeleteBuffers(1, [self.hair_vbo])
                gl.glDeleteBuffers(1, [self.hair_ebo])
            except Exception:
                pass

        # VAO
        self.hair_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.hair_vao)

        # VBO
        self.hair_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.hair_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.hair_vertex_data.nbytes,
            self.hair_vertex_data,
            gl.GL_DYNAMIC_DRAW,
        )

        # Vec4 Positions
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))

        # EBO
        self.hair_ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.hair_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self.hair_index_data.nbytes,
            self.hair_index_data,
            gl.GL_STATIC_DRAW,
        )

        gl.glPrimitiveRestartIndex(self.restart_index)
        gl.glEnable(gl.GL_PRIMITIVE_RESTART)

        gl.glBindVertexArray(0)

    def update_buffers(self, hair_system):
        """
        Updates the hair centerline VBO with latest strand positions.
        Keeps the existing index buffer.

        Args:
            hair_system (HairSystem): The hair system containing strands to render.
        """
        index = 0
        for strand in hair_system.strands:
            n = strand.positions_np.shape[0]
            packed = np.hstack([strand.positions_np, strand.radius_np[:, None]]).astype(
                np.float32
            )
            self.hair_vertex_data[index : index + n] = packed
            index += n

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.hair_vbo)
        gl.glBufferSubData(
            gl.GL_ARRAY_BUFFER, 0, self.hair_vertex_data.nbytes, self.hair_vertex_data
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def render(self, camera, color, hair_style):
        """
        Renders the hair strands using the current buffers and shader settings.

        Args:
            camera (Camera): The camera used for rendering.
            color (Vector3): The color to render the hair.
            hair_style (int): The style of hair to render (0 for lines, 2 for points).
        """
        if self.hair_vao is None:
            return

        hair_tx = Transform()
        ShaderLib.set_uniform("MVP", camera.get_vp() @ hair_tx.get_matrix())
        ShaderLib.set_uniform("color", color.x, color.y, color.z)

        gl.glBindVertexArray(self.hair_vao)

        # Render lines
        ShaderLib.set_uniform("render_points", 0)
        gl.glDrawElements(
            gl.GL_LINE_STRIP,
            int(self.hair_num_indices),
            gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0),
        )

        # Render braids (optional)
        if hair_style == 2:
            ShaderLib.set_uniform("render_points", 1)
            gl.glDrawElements(
                gl.GL_POINTS,
                int(self.hair_num_indices),
                gl.GL_UNSIGNED_INT,
                ctypes.c_void_p(0),
            )

        gl.glBindVertexArray(0)

    def build_cpu_data(self, hair_system):
        """
        Pure CPU packing.
        """
        vertices = []
        indices = []
        index = 0

        for strand in hair_system.strands:
            n = strand.positions_np.shape[0]
            packed = np.hstack([strand.positions_np, strand.radius_np[:, None]]).astype(np.float32)

            vertices.extend(packed.tolist())
            indices.extend(range(index, index + n))
            index += n
            indices.append(self.restart_index)

        vertex_data = np.asarray(vertices, dtype=np.float32).reshape(-1, 4)
        index_data = np.asarray(indices, dtype=np.uint32)
        return vertex_data, index_data
