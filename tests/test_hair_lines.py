import numpy as np

from render.hair_line_renderer import HairLineRenderer
from tests.helpers import make_simple_hair_system

def test_build_cpu_data():
    hs = make_simple_hair_system()
    render = HairLineRenderer()

    vector, index = render.build_cpu_data(hs)

    assert vector.shape == (3, 4)
    assert index.tolist() == [0, 1, 2, render.restart_index]

def test_build_cpu_data_packs():
    hs = make_simple_hair_system()
    strand = hs.strands[0]
    strand.radius_np[:] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    render = HairLineRenderer()
    vector, _ = render.build_cpu_data(hs)
    
    assert np.allclose(vector[:, 3], strand.radius_np)