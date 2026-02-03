import numpy as np
from ncca.ngl.vec3 import Vec3
from module.hair_particle import HairParticle
from module.hair_strand import HairStrand
from module.hair_system import HairSystem

def make_simple_hair_system():
    p1 = HairParticle(Vec3(0,0,0), Vec3(0,0,0), pinned=True, pin_position=Vec3(0,0,0))
    p2 = HairParticle(Vec3(0,-1,0), Vec3(0,-1,0))
    p3 = HairParticle(Vec3(0,-2,0), Vec3(0,-2,0))

    strand = HairStrand(
        particles=[p1,p2,p3],
        root_position=Vec3(0,0,0),
        _segment_length=1.0
    )

    hs = HairSystem()
    hs.add_strand(strand)
    
    return hs
