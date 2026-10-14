"""Geometry parameters of the pre-demonstrator components (lengths in mm)."""
from dataclasses import dataclass
from functools import partial

import numpy as np

from .constants import (AEROGEL_HALF, AEROGEL_T, BOARD_EDGE, BOARD_NPAD, BOARD_PACKAGE, BOARD_PAD, MIRROR_CLEAR_FRAC,
                        MIRROR_REFL_NOMINAL, MIRROR_REFL_NOMINAL_NM, PIXEL_PITCH, T_ROOM, WINDOW_N, WINDOW_T)
from .media import XE_INDEX_DEFAULT, ConstantIndex, Medium, Rayleigh, aerogel_constant, n_xe
from .tracer import Scene, SensorPlane, Slab, SphereMirror


MIRROR_REFLECTIVITIES = ('1', 'nominal')    # named reflectivity options; a number in (0, 1] gives a flat reflectivity


def _flat(lam_nm, R):
    return np.full(np.shape(lam_nm), R)


def mirror_reflectivity(spec=None):
    """Mirror reflectivity R(lambda [nm]) of an option: None, 1 or '1' -> None (reflectivity 1, today's default); 'nominal'
    -> MIRROR_REFL_NOMINAL (an assumed curve from the vendor spec, not a measured one), linear in wavelength between its
    nodes and held beyond them; a number in (0, 1] (or its string) -> that flat value."""
    if spec is None or spec == 'nominal':
        return None if spec is None else partial(np.interp, xp=MIRROR_REFL_NOMINAL_NM, fp=MIRROR_REFL_NOMINAL)
    R = float(spec)
    if not 0 < R <= 1:
        raise ValueError('mirror reflectivity %r outside (0, 1]' % spec)
    return None if R == 1 else partial(_flat, R=R)


@dataclass(frozen=True)
class Mirror:
    """Spherical mirror: radius of curvature R, diameter D, clear aperture fraction, reflectivity option (mirror_reflectivity;
    None: 1)."""
    R: float
    D: float | None = None
    clear_frac: float = MIRROR_CLEAR_FRAC
    reflectivity: str | float | None = None

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
class Board:
    """Pixelated sensor centred on the axis: 2 x 2 abutting square packages of side package, each with npad x npad active
    pads of side pad at pitch, the first pad edge from the package edge (the ARC prototype board of S13361-3050NE-08
    arrays by default)."""
    package: float = BOARD_PACKAGE
    edge: float = BOARD_EDGE
    pad: float = BOARD_PAD
    npad: int = BOARD_NPAD
    pitch: float = PIXEL_PITCH

    @property
    def half(self):
        """Half side of the footprint (package outline)."""
        return self.package

    @property
    def npads(self):
        """Number of pads."""
        return (2*self.npad)**2

    @property
    def pad_lo(self):
        """Low edges of the 2 npad pad columns (or rows) along one axis [mm], increasing."""
        k = np.arange(self.npad); pos = self.edge+self.pitch*k
        return np.r_[-(pos[::-1]+self.pad), pos]

    def _axis(self, x):
        lo = self.pad_lo; i = np.searchsorted(lo, x, 'right')-1; ic = np.clip(i, 0, len(lo)-1)
        return np.where((i >= 0)&(x < lo[ic]+self.pad), i, -1)

    def pad_index(self, x, y):
        """Pad index (column x 2 npad + row) of the points (x, y) [mm], -1 outside the active pads."""
        ix = self._axis(np.asarray(x, float)); iy = self._axis(np.asarray(y, float))
        return np.where((ix >= 0)&(iy >= 0), ix*2*self.npad+iy, -1)

    def fill(self, half=None):
        """Active fraction of the square |x|, |y| < half (default the footprint)."""
        half = self.half if half is None else half
        lo = self.pad_lo; a = np.clip(np.minimum(lo+self.pad, half)-np.maximum(lo, -half), 0, None).sum()
        return float((a/(2*half))**2)


SENSOR_MODELS = ('square', 'board')        # continuous active square; Board pixel map with binary pad hits


@dataclass(frozen=True)
class Sensor:
    """Photosensor centred on the axis: half side and pixel pitch of a continuous active square (model 'square'), or the
    Board pixel map (model 'board': the sensor plane is the board footprint, half is not used)."""
    half: float
    pitch: float = PIXEL_PITCH
    model: str = 'square'

    def __post_init__(self):
        if self.model not in SENSOR_MODELS:
            raise ValueError('unknown sensor model %r (choose from %s)' % (self.model, ', '.join(SENSOR_MODELS)))

    @property
    def board(self):
        """Board of the 'board' model, else None."""
        return Board() if self.model == 'board' else None

    @property
    def plane_half(self):
        """Half side of the sensor plane that collects photons: the board footprint or half."""
        return self.board.half if self.model == 'board' else self.half


@dataclass(frozen=True)
class Aerogel:
    """Square aerogel tile centred on the axis: refractive index, half side, thickness."""
    n: float
    half: float = AEROGEL_HALF
    thickness: float = AEROGEL_T


@dataclass(frozen=True)
class Window:
    """SiPM protective window covering the sensor, directly below the sensor plane: refractive index, thickness."""
    n: float = WINDOW_N
    thickness: float = WINDOW_T

    @property
    def medium(self):
        return Medium('window n = %g' % self.n, ConstantIndex(self.n))


@dataclass(frozen=True)
class Radiator:
    """Xe gas radiator: pressure [bar], length along the axis, temperature [K], n_xe index model."""
    P_bar: float
    length: float | None = None
    T: float = T_ROOM
    model: str = XE_INDEX_DEFAULT

    def n(self, lam_nm):
        return n_xe(lam_nm, self.P_bar, self.T, self.model)


@dataclass(frozen=True)
class Phase1Layout:
    """Mirror setup of configurations 1 and 5 along the axis (tile optional), z up from the mirror vertex (lengths in mm).

    Spherical mirror (vertex at z = 0, centre at z = R), gas, aerogel tile (None: no tile) of
    half side tile.half between z_tile_bottom and z_tile_top, a gas gap, and the sensor plane at
    z_sensor, perpendicular to and centred on the axis; z_sensor_ overrides its default, the
    paraxial focal plane z = f. The whole volume is filled with the gas; tile_medium defaults to
    a constant index tile.n. tile_scatter (media.Rayleigh or None) adds Rayleigh scattering in the
    tile. window (Window or None) puts a window of the sensor size in the gap, directly below the
    sensor plane, whose top face absorbs like the sensor.
    """
    mirror: Mirror
    sensor: Sensor
    gas: Medium
    tile: Aerogel | None = None
    gap: float = 1.0
    tile_medium: Medium | None = None
    z_sensor_: float | None = None
    tile_scatter: Rayleigh | None = None
    window: Window | None = None

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
    def z_window_bottom(self):
        return self.z_sensor-self.window.thickness

    @property
    def tile_index(self):
        """Medium of the tile."""
        return self.tile_medium if self.tile_medium is not None else aerogel_constant(self.tile.n)

    def scene(self):
        """tracer.Scene of the layout: gas, tile slab (then window slab, if any), mirror cap and sensor plane."""
        if self.gap <= 0:
            raise ValueError('the gap between tile and sensor must be positive')
        slabs = ()
        if self.tile is not None:
            h = self.tile.half
            slabs = (Slab((-h, -h, self.z_tile_bottom), (h, h, self.z_tile_top), self.tile_index, self.tile_scatter),)
        if self.window is not None:
            if self.tile is not None and self.window.thickness >= self.gap:
                raise ValueError('the window must be thinner than the gap')
            h = self.sensor.plane_half
            slabs += (Slab((-h, -h, self.z_window_bottom), (h, h, self.z_sensor), self.window.medium),)
        mirror = SphereMirror((0.0, 0.0, self.mirror.R), self.mirror.R, (0.0, 0.0, 1.0), self.mirror.clear_radius,
                              mirror_reflectivity(self.mirror.reflectivity))
        return Scene(self.gas, slabs, mirror, SensorPlane(self.z_sensor, self.sensor.plane_half))
