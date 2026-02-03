from dataclasses import dataclass, field
from typing import Optional

from ncca.ngl.vec3 import Vec3


@dataclass
class HairParticle:
    position_init: Vec3
    _prev_position_init: Vec3
    acceleration: Vec3 = field(default_factory=Vec3)
    mass: float = 1.0
    damping: float = 0.99
    pinned: bool = False
    pin_position: Optional[Vec3] = None
    length: float = 1.0
    _index: Optional[int] = None
    _strand: Optional["HairStrand"] = None
    _local_position: Vec3 = None
    _local_prev: Vec3 = None

    def __post_init__(self):
        # Backup the initial positions
        self._local_position = self.position_init
        self._local_prev = self._prev_position_init

    @property
    def position(self):
        if self._strand is None:
            return self._local_position
        x, y, z = self._strand.positions_np[self._index]
        return Vec3(x, y, z)

    @position.setter
    def position(self, v: Vec3):
        if self._strand is None:
            self._local_position = v
        else:
            self._strand.positions_np[self._index] = (v.x, v.y, v.z)

    @property
    def prev_position(self):
        if self._strand is None:
            return self._local_prev
        x, y, z = self._strand.prev_positions_np[self._index]
        return Vec3(x, y, z)

    @prev_position.setter
    def prev_position(self, v: Vec3):
        if self._strand is None:
            self._local_prev = v
        else:
            self._strand.prev_positions_np[self._index] = (v.x, v.y, v.z)

    def attach(self, strand, index):
        """
        Attaches the particle to a strand for vectorized operations.

        Args:
            strand (HairStrand): The hair strand to attach to.
            index (int): The index of the particle within the strand.
        """
        self._strand = strand
        self._index = index
