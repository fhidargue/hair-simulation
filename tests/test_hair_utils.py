import numpy as np
import pytest

from PIL import Image
from ncca.ngl import Vec3

from lib.utils import (
    _euler_xyz_matrix_deg,
    apply_face_transform,
    rotate_direction,
    load_grayscale_mask,
    ngl_mat4_to_np,
)

class FakeMat4:
    def __init__(self, row_major_4x4: np.ndarray):
        self._m = np.asarray(row_major_4x4, dtype=np.float32)

    def __getitem__(self, c):
        col = self._m[:, c].copy()

        class Col:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, r):
                return float(self.data[r])

        return Col(col)

# Helper functions
def assert_mat_close(A, B, atol=1e-5, rtol=1e-5):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    assert A.shape == B.shape
    np.testing.assert_allclose(A, B, atol=atol, rtol=rtol)


def assert_vec_close(a, b, atol=1e-5, rtol=1e-5):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def test_euler_matrix_identity():
    R = _euler_xyz_matrix_deg(Vec3(0, 0, 0))
    assert R.dtype == np.float32
    assert_mat_close(R, np.eye(3, dtype=np.float32))


@pytest.mark.parametrize(
    "rotate, expected",
    [
        (Vec3(90, 0, 0), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)),
        (Vec3(0, 90, 0), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)),
        (Vec3(0, 0, 90), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)),
    ],
)
def test_euler_matrix_axis_rotate(rotate, expected):
    R = _euler_xyz_matrix_deg(rotate)
    assert_mat_close(R, expected, atol=2e-5, rtol=2e-5)


def test_euler_matrix_orthonormal():
    # random angles
    R = _euler_xyz_matrix_deg(Vec3(23.0, -41.0, 137.0))
    I = np.eye(3, dtype=np.float32)

    assert_mat_close(R.T @ R, I, atol=2e-4, rtol=2e-4)

    determinant = float(np.linalg.det(R.astype(np.float64)))
    assert determinant == pytest.approx(1.0, abs=5e-4)


def test_rotate_direction_90():
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = rotate_direction(direction, Vec3(0, 0, 90))
    assert_vec_close(out, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=2e-5)


def test_rotate_direction_length():
    direction = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    out = rotate_direction(direction, Vec3(12.0, 34.0, 56.0))

    n0 = float(np.linalg.norm(direction.astype(np.float64)))
    n1 = float(np.linalg.norm(out.astype(np.float64)))
    assert n1 == pytest.approx(n0, rel=1e-5, abs=1e-5)


def test_apply_face_transform():
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = apply_face_transform(vector, position=Vec3(0, 0, 0), scale=1.0, rotation=Vec3(0, 0, 0))
    assert_vec_close(out, vector)


def test_apply_face_transform_with_translation():
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    position = Vec3(10, -5, 2)
    out = apply_face_transform(vector, position=position, scale=1.0, rotation=Vec3(0, 0, 0))
    assert_vec_close(out, np.array([11.0, -3.0, 5.0], dtype=np.float32))


def test_apply_face_transform_with_scale():
    vector = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    out = apply_face_transform(vector, position=Vec3(0, 0, 0), scale=2.0, rotation=Vec3(0, 0, 0))
    assert_vec_close(out, np.array([2.0, -4.0, 1.0], dtype=np.float32))


def test_apply_face_transform_rotation_z_90_then_translate():
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = apply_face_transform(vector, position=Vec3(5, 6, 7), scale=1.0, rotation=Vec3(0, 0, 90))
    assert_vec_close(out, np.array([5.0, 7.0, 7.0], dtype=np.float32), atol=2e-5)


def test_load_grayscale_mask(tmp_path):
    image = Image.fromarray(
        np.array([[0, 255], [128, 64]], dtype=np.uint8),
        mode="L",
    )
    p = tmp_path / "mask.png"
    image.save(p)

    mask = load_grayscale_mask(str(p))
    assert mask.dtype == np.float32
    assert mask.shape == (2, 2)

    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0

    assert mask[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert mask[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert mask[1, 0] == pytest.approx(128.0 / 255.0, abs=1e-6)
    assert mask[1, 1] == pytest.approx(64.0 / 255.0, abs=1e-6)


def test_ngl_mat4_to_np():
    random_matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.float32,
    )
    matrix = FakeMat4(random_matrix)
    out = ngl_mat4_to_np(matrix)

    assert out.dtype == np.float32
    assert out.shape == (4, 4)
    assert_mat_close(out, random_matrix)
