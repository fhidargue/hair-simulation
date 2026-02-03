import numpy as np
from render.hair_ribbon_renderer import HairRibbonRenderer
from tests.helpers import make_simple_hair_system


def test_tbo_segment_count():
    hs = make_simple_hair_system()
    renderer = HairRibbonRenderer()

    seg_count, tbo = renderer.build_cpu_tbo_data(hs)

    assert seg_count == 2
    assert tbo.shape == (4, 4)


def test_tbo_packing_A_B():
    hs = make_simple_hair_system()
    strand = hs.strands[0]
    strand.radius_np[:] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    renderer = HairRibbonRenderer()
    _, tbo = renderer.build_cpu_tbo_data(hs)

    # segment 0
    assert np.allclose(tbo[0, 0:3], strand.positions_np[0])
    assert np.allclose(tbo[1, 0:3], strand.positions_np[1])
    assert tbo[0, 3] == np.float32(0.1)
    assert tbo[1, 3] == np.float32(0.2)

    # segment 1
    assert np.allclose(tbo[2, 0:3], strand.positions_np[1])
    assert np.allclose(tbo[3, 0:3], strand.positions_np[2])
    assert tbo[2, 3] == np.float32(0.2)
    assert tbo[3, 3] == np.float32(0.3)
