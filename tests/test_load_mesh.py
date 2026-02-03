from lib.mesh_loader import load_mesh

from lib.constants import FACE_MESH_PATH


def test_load_obj():
    vertices, normals, indices, uvs = load_mesh(FACE_MESH_PATH)

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert normals.ndim == 2 and normals.shape[1] == 3

    assert indices.ndim == 1
    assert indices.size > 0
    assert indices.size % 3 == 0

    if uvs is not None:
        assert uvs.ndim == 2 and uvs.shape[1] == 2

