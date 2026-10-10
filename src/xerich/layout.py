"""Geometry parameters of the demonstrator components (lengths in mm)."""
from dataclasses import dataclass

from .constants import AEROGEL_HALF, AEROGEL_T, MIRROR_CLEAR_FRAC, PIXEL_PITCH, T_ROOM
from .media import n_xe


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
