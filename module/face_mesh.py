import ctypes

import numpy as np
import OpenGL.GL as gl

from lib.mesh_loader import load_mesh


class FaceMesh:
    def __init__(self, path: str, is_testing: bool = False):
        """
        Initializes the face mesh by loading it from the specified file path.

        Args:
            path (str): The file path to the 3D model of the face mesh.
            is_testing (bool): This property determines if the function is for testing or not.
        """
        mesh_data = load_mesh(path)

        if len(mesh_data) == 3:
            vertices, normals, indices = mesh_data
            uvs = None
        elif len(mesh_data) == 4:
            vertices, normals, indices, uvs = mesh_data
        else:
            raise RuntimeError(
                f"load_mesh() returned {len(mesh_data)} items, expected 3 or 4."
            )

        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        self.uvs = None if uvs is None else np.ascontiguousarray(uvs, dtype=np.float32)

        self.indices = np.ascontiguousarray(indices.reshape(-1), dtype=np.uint32)
        self.index_count = int(self.indices.size)

        self.vertex_data = np.ascontiguousarray(
            np.hstack([self.vertices, self.normals]),
            dtype=np.float32,
        )

        if not is_testing:
            self._create_face_gpu_buffers()

    def _create_face_gpu_buffers(self):
        """
        Creates the GPU buffers (VAO, VBO, EBO) for rendering the face mesh.
        """
        # VAO
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # VBO
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.vertex_data.nbytes,
            self.vertex_data,
            gl.GL_STATIC_DRAW,
        )

        stride = 6 * 4  # 6 floats

        # VBO position
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)
        )

        # VBO normal
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12)
        )

        # EBO
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self.indices.nbytes,
            self.indices,
            gl.GL_STATIC_DRAW,
        )

        gl.glBindVertexArray(0)

    def draw(self):
        """
        Renders the face mesh using the previously created GPU buffers.
        """
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.index_count,
            gl.GL_UNSIGNED_INT,
            ctypes.c_void_p(0),
        )
        gl.glBindVertexArray(0)
