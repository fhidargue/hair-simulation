import numpy as np
from ncca.ngl.vec3 import Vec3

from module.hair_particle import HairParticle
from module.hair_strand import HairStrand


def test_particle_attach():
    particle = HairParticle(
        position_init=Vec3(1.0, 2.0, 3.0),
        _prev_position_init=Vec3(1.0, 2.0, 3.0),
    )

    assert particle.position == Vec3(1.0, 2.0, 3.0)

    strand = HairStrand(
        particles=[particle],
        root_position=Vec3(0, 0, 0),
        _segment_length=1.0,
        _stiffness=1.0,
    )

    assert particle._strand is strand
    assert particle._index == 0

    np_position = strand.positions_np[0]
    assert np.allclose(np_position, [1.0, 2.0, 3.0])

    particle.position = Vec3(9.0, 8.0, 7.0)

    assert np.allclose(strand.positions_np[0], [9.0, 8.0, 7.0])
    assert particle.position == Vec3(9.0, 8.0, 7.0)
