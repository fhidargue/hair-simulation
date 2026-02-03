import numpy as np
import pytest
from ncca.ngl.vec3 import Vec3

from lib.constants import FACE_MESH_PATH
from lib.hair_factory import create_head_hair_from_mesh
from lib.utils import make_strand_test
from module.face_mesh import FaceMesh
from module.hair_system import HairSystem


@pytest.fixture(scope="module")
def face_mesh():
    return FaceMesh(FACE_MESH_PATH, True)


def test_error_create_head_hair():
    hair_system = HairSystem()

    class EmptyMesh:
        vertices = np.array([])
        normals = np.array([])

    with pytest.raises(RuntimeError):
        create_head_hair_from_mesh(
            hair_system=hair_system,
            face_mesh=EmptyMesh(),
            face_position=Vec3(0, 0, 0),
            face_scale=1.0,
            face_rotation=Vec3(0, 0, 0),
            strands=10,
            spacing=0.1,
            particles=5,
            make_strand_func=make_strand_test,
            is_testing=True,
        )


def test_create_head_hair_from_mesh_strand_count(face_mesh):
    hair_system = HairSystem()

    requested = 50

    create_head_hair_from_mesh(
        hair_system=hair_system,
        face_mesh=face_mesh,
        face_position=Vec3(0, 0, 0),
        face_scale=1.0,
        face_rotation=Vec3(0, 0, 0),
        strands=requested,
        spacing=0.05,
        particles=10,
        make_strand_func=make_strand_test,
        is_testing=True,
    )

    assert len(hair_system.strands) > 0
    assert len(hair_system.strands) <= requested


def test_create_head_hair_particles(face_mesh):
    hair_system = HairSystem()
    particles = 3

    create_head_hair_from_mesh(
        hair_system=hair_system,
        face_mesh=face_mesh,
        face_position=Vec3(0, 0, 0),
        face_scale=1.0,
        face_rotation=Vec3(0, 0, 0),
        strands=10,
        spacing=0.02,
        particles=particles,
        make_strand_func=make_strand_test,
        is_testing=True,
    )

    for strand in hair_system.strands:
        assert len(strand.particles) == particles


def test_create_head_hair_from_mesh_pinned(face_mesh):
    hair_system = HairSystem()

    create_head_hair_from_mesh(
        hair_system=hair_system,
        face_mesh=face_mesh,
        face_position=Vec3(0, 0, 0),
        face_scale=1.0,
        face_rotation=Vec3(0, 0, 0),
        strands=5,
        spacing=0.05,
        particles=6,
        make_strand_func=make_strand_test,
        is_testing=True,
    )

    for strand in hair_system.strands:
        assert strand.particles[0].pinned is True


def test_create_head_hair_from_not_pinned(face_mesh):
    hair_system = HairSystem()

    create_head_hair_from_mesh(
        hair_system=hair_system,
        face_mesh=face_mesh,
        face_position=Vec3(0, 4.0, 0),
        face_scale=1.0,
        face_rotation=Vec3(0, 0, 0),
        strands=20,
        spacing=0.05,
        particles=8,
        make_strand_func=make_strand_test,
        is_testing=True,
    )

    assert len(hair_system.strands) > 0
    assert len(hair_system.strands) <= 20

    for strand in hair_system.strands:
        assert strand.positions_np.shape[0] > 0
        assert strand.pinned_np[0] == True
