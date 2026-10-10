"""Vectorised 3D ray geometry.

Points and directions are arrays of shape (..., 3) with z as the optical axis;
directions are unit vectors.
"""
import numpy as np

ORIGIN = np.zeros(3)


def basis(u):
    """Two unit vectors e1, e2 orthogonal to each direction in u (N,3), with e2 = u x e1."""
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0, 0, 1.]]), np.array([[1., 0, 0]]))
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(u, e1)
    return e1, e2


def cone(u, th, phi, e1=None, e2=None):
    """Directions at angle th around each u (N,3) for azimuths phi (M,), shape (N,M,3).

    The azimuth is measured from e1 towards e2 (default: basis(u)).
    """
    if e1 is None:
        e1, e2 = basis(u)
    return (np.cos(th)*u[:, None, :]+np.sin(th)*(np.cos(phi)[None, :, None]*e1[:, None, :]
                                                 + np.sin(phi)[None, :, None]*e2[:, None, :]))


def refract(d, N, eta):
    """Snell refraction of directions d at a surface with unit normal N facing the incoming ray.

    eta = n_in/n_out. Returns the refracted unit directions and a mask that is False
    for total internal reflection.
    """
    cosi = -(d*N).sum(-1, keepdims=True); k = 1-eta**2*(1-cosi**2)
    ok = (k[..., 0] > 0); t = eta*d+(eta*cosi-np.sqrt(np.clip(k, 0, None)))*N
    return t/np.linalg.norm(t, axis=-1, keepdims=True), ok


def reflect(d, n):
    """Specular reflection of directions d at surfaces with unit normal n."""
    return d-2*(d*n).sum(-1, keepdims=True)*n


def sphere_far(p, d, C, R):
    """Far intersection of the rays p + s d with the sphere |x - C| = R.

    Returns the points and a mask of the rays that meet the sphere.
    """
    m = p-C; b = (m*d).sum(-1); c = (m*m).sum(-1)-R*R; disc = b*b-c
    s = -b+np.sqrt(np.clip(disc, 0, None))
    return p+s[..., None]*d, disc > 0


def sphere_normal(q, C, R):
    """Outward unit normal of the sphere |x - C| = R at the points q."""
    return (q-C)/R


def to_plane(p, d, z, valid=None):
    """Points where the rays p + t d cross the plane at height z.

    Rays outside the optional mask `valid` use d_z = 1 to avoid dividing by zero.
    """
    dz = d[..., 2] if valid is None else np.where(valid, d[..., 2], 1)
    t = (z-p[..., 2])/dz
    return p+t[..., None]*d


def advance(p, d, dz):
    """Move the points p along d by the height difference dz."""
    t = dz/d[..., 2]
    return p+t[..., None]*d


def snell_angle(th, n1, n2):
    """Angle to the normal after crossing a plane interface from index n1 to n2."""
    return np.arcsin(n1*np.sin(th)/n2)
