import numpy as np
import pytest
from ncca.ngl.vec3 import Vec3

from lib.utils import make_strand_test


def test_apply_global_forces():
    strand = make_strand_test()

    gravity = Vec3(0.0, -9.81, 0.0)
    wind = Vec3(1.0, 0.0, 0.0)
    expected = np.array(
        [gravity.x + wind.x, gravity.y + wind.y, gravity.z + wind.z], dtype=np.float32
    )

    strand.apply_global_forces(gravity, wind)

    # Root pinned
    assert np.allclose(strand.acc_np[0], np.zeros(3))

    for i in range(1, len(strand.particles)):
        assert np.allclose(strand.acc_np[i], expected)


def test_enforce_constraints_root_fixed():
    strand = make_strand_test()
    root = strand.particles[0]

    original = np.array([root.position.x, root.position.y, root.position.z])

    strand.enforce_constraints()

    assert np.allclose(strand.positions_np[0], original)
    assert np.allclose(strand.prev_positions_np[0], original)


def test_enforce_constraints_corrects_length():
    strand = make_strand_test()

    distance = np.linalg.norm(strand.positions_np[1] - strand.positions_np[0])
    assert distance == pytest.approx(1.0, abs=0.10)

    for _ in range(len(strand.particles)):
        strand.enforce_constraints()

    distance_2 = np.linalg.norm(strand.positions_np[2] - strand.positions_np[1])
    assert distance_2 == pytest.approx(1.0, abs=0.10)


def test_enforce_constraints_multiple_iterations():
    strand = make_strand_test()

    for _ in range(10):
        strand.enforce_constraints()

    distance = np.linalg.norm(strand.positions_np[1] - strand.positions_np[0])
    distance_2 = np.linalg.norm(strand.positions_np[2] - strand.positions_np[1])

    assert distance == pytest.approx(1.0, abs=0.10)
    assert distance_2 == pytest.approx(1.0, abs=0.10)


def test_apply_global_forces_pin():
    strand = make_strand_test()
    gravity = Vec3(0.0, -9.81, 0.0)
    wind = Vec3(1.0, 0.0, 0.0)
    expected = np.array([1.0, -9.81, 0.0], dtype=np.float32)

    strand.apply_global_forces(gravity, wind)
    assert np.allclose(strand.acc_np[0], np.zeros(3, dtype=np.float32))

    for i in range(1, len(strand.particles)):
        assert np.allclose(strand.acc_np[i], expected)


def test_enforce_constraints_lengths_close():
    strand = make_strand_test()

    for _ in range(10):
        strand.enforce_constraints()

    segment = strand.positions_np[1:] - strand.positions_np[:-1]
    lengths = np.linalg.norm(segment, axis=1)

    np.testing.assert_allclose(lengths, strand.rest_lengths_np, atol=0.10, rtol=0.0)


def test_curly_hair():
    strand = make_strand_test()
    strand.is_curly = True
    strand.__post_init__()

    np.testing.assert_allclose(
        strand.positions_np,
        strand.rest_positions_np,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        strand.prev_positions_np,
        strand.rest_positions_np,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        strand.acc_np,
        np.zeros_like(strand.acc_np),
    )

    assert np.linalg.norm(strand.curl_offset_np[1:]) > 0.0


def test_update_calls_collision_callback():
    strand = make_strand_test()
    called = {"n": 0}

    def collide_fn(s):
        assert s is strand
        called["n"] += 1

    strand.update(0.016, collide_fn=collide_fn)
    assert called["n"] == 1