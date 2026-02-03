import numpy as np
import trimesh


def load_mesh(path: str):
    """
    Load a 3D mesh from the specified file path using trimesh.

    Args:
        path (str): The file path to the 3D mesh.
    """
    scene = trimesh.load(path, force="scene")

    if isinstance(scene, trimesh.Scene):
        if not scene.geometry:
            raise ValueError(f"No valid geometry found for the path: {path}")

        meshes = []
        for geometry in scene.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        if not meshes:
            raise ValueError(f"No valid meshes found for path: {path}")

        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    # Ensure normals
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.merge_vertices()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.rezero()
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    indices = np.asarray(mesh.faces.flatten(), dtype=np.uint32)

    uvs = None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)

    return vertices, normals, indices, uvs
