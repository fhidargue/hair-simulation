import numpy as np

from lib.utils import make_strand_test
from module.hair_system import HairSystem


class DummyFace:
    def __init__(self, verts):
        self.vertices = np.asarray(verts, dtype=np.float32)


def test_add_strand():
    hair_system = HairSystem()
    strand = make_strand_test()

    hair_system.add_strand(strand)

    assert len(hair_system.strands) == 1
    assert hair_system.strands[0] is strand


def test_gravity_on_particles():
    hair_system = HairSystem()
    strand = make_strand_test()
    hair_system.add_strand(strand)

    before = strand.positions_np[1, 1]

    hair_system.update(0.1)

    after = strand.positions_np[1, 1]

    assert after < before

def test_add_strand():
    hs = HairSystem()
    s = make_strand_test()
    hs.add_strand(s)
    assert hs.strands == [s]


def test_gravity_to_particles():
    hs = HairSystem()
    s = make_strand_test()
    hs.add_strand(s)

    y_before = float(s.positions_np[1, 1])
    hs.update(0.1)
    y_after = float(s.positions_np[1, 1])

    assert y_after < y_before


def test_set_colliders():
    hs = HairSystem()
    face = DummyFace(
        verts=[
            [0, 0, 0],
            [2, 4, 6],
        ]
    )

    hs.set_collider_from_face_mesh(face, shrink=1.0)

    np.testing.assert_allclose(hs.collider_center_local_np, np.array([1, 2, 3], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(hs.collider_radii_local_np,  np.array([1, 2, 3], dtype=np.float32), atol=1e-6)

    # Nose/mouth colliders should be non-zero radii and finite centers
    assert hs.nose_radii_local_np.shape == (3,)
    assert hs.mouth_radii_local_np.shape == (3,)
    assert np.all(np.isfinite(hs.nose_center_local_np))
    assert np.all(np.isfinite(hs.mouth_center_local_np))
    assert np.all(hs.nose_radii_local_np > 0)
    assert np.all(hs.mouth_radii_local_np > 0)


def test_collide_with_ellipsoid():
    hs = HairSystem()
    hs.set_head_matrices(np.eye(4, dtype=np.float32))

    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    radii  = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    s = make_strand_test()
    hs.add_strand(s)

    # Slightly inside position
    s.positions_np[1] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    s.prev_positions_np[1] = s.positions_np[1].copy()

    moved = hs._collide_with_ellipsoid(s, center, radii)
    assert moved >= 1

    p = s.positions_np[1]
    q = (p - center) / (radii + 1e-8)
    d2 = float(np.dot(q, q))
    assert d2 >= 1.0 - 1e-3
