"""Geometry parameters of the demonstrator components (lengths in mm)."""
from dataclasses import dataclass

from .constants import AEROGEL_HALF, AEROGEL_T, MIRROR_CLEAR_FRAC, PIXEL_PITCH, T_ROOM
from .media import Medium, aerogel_constant, n_xe
from .tracer import Scene, SensorPlane, Slab, SphereMirror


@dataclass(frozen=True)
class Mirror:
    """Spherical mirror: radius of curvature R, diameter D, clear aperture fraction."""
    R: float
    D: float | None = None
    clear_frac: float = MIRROR_CLEAR_FRAC

    @classmethod
    def from_focal(cls, f, D=None, **kw):
        return cls(2*f, D, **kw)

    @property
    def f(self):
        """Paraxial focal length R/2."""
        return self.R/2

    @property
    def clear_radius(self):
        """Radius of the usable aperture."""
        return self.clear_frac*self.D/2


@dataclass(frozen=True)
class Sensor:
    """Square photosensor centred on the axis: half side and pixel pitch."""
    half: float
    pitch: float = PIXEL_PITCH


@dataclass(frozen=True)
class Aerogel:
    """Square aerogel tile centred on the axis: refractive index, half side, thickness."""
    n: float
    half: float = AEROGEL_HALF
    thickness: float = AEROGEL_T


@dataclass(frozen=True)
class Radiator:
    """Xe gas radiator: pressure [bar], length along the axis, temperature [K]."""
    P_bar: float
    length: float | None = None
    T: float = T_ROOM

    def n(self, lam_nm):
        return n_xe(lam_nm, self.P_bar, self.T)


@dataclass(frozen=True)
class Phase1Layout:
    """Phase-1 mirror setup along the axis, z up from the mirror vertex (lengths in mm).

    Spherical mirror (vertex at z = 0, centre at z = R), gas, aerogel tile (None: no tile) of
    half side tile.half between z_tile_bottom and z_tile_top, a gas gap, and the sensor plane at
    z_sensor, perpendicular to and centred on the axis; z_sensor_ overrides its default, the
    paraxial focal plane z = f. The whole volume is filled with the gas; tile_medium defaults to
    a constant index tile.n.
    """
    mirror: Mirror
    sensor: Sensor
    gas: Medium
    tile: Aerogel | None = None
    gap: float = 1.0
    tile_medium: Medium | None = None
    z_sensor_: float | None = None

    @property
    def z_sensor(self):
        return self.mirror.f if self.z_sensor_ is None else self.z_sensor_

    @property
    def z_tile_top(self):
        return self.z_sensor-self.gap

    @property
    def z_tile_bottom(self):
        return self.z_tile_top-self.tile.thickness

    @property
    def tile_index(self):
        """Medium of the tile."""
        return self.tile_medium if self.tile_medium is not None else aerogel_constant(self.tile.n)

    def scene(self):
        """tracer.Scene of the layout: gas, tile slab, mirror cap and sensor plane."""
        if self.gap <= 0:
            raise ValueError('the gap between tile and sensor must be positive')
        slabs = ()
        if self.tile is not None:
            h = self.tile.half
            slabs = (Slab((-h, -h, self.z_tile_bottom), (h, h, self.z_tile_top), self.tile_index),)
        mirror = SphereMirror((0.0, 0.0, self.mirror.R), self.mirror.R, (0.0, 0.0, 1.0), self.mirror.clear_radius)
        return Scene(self.gas, slabs, mirror, SensorPlane(self.z_sensor, self.sensor.half))
