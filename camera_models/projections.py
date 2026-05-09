import numpy as np  
import torch


class ArctanProjection:
    """
    Arctan projection: uv = xy / |xy| * atan2(|xy|, z)
    Implements an equidistant azimuthal (arctan) projection between 3D points and 2D planar coordinates
    
    This is what the Oculus fisheye models use: the length of the 2D
    u vector is the angle in radians between +Z and the target ray
    direction.

    Unlike perspective projection, this can represent a FOV over 180
    degrees. However, distortion increases as the FOV gets wider. Near
    180 degrees, the aspect ratio of a pixel is about 1:1.57, which
    must then be corrected for in the distortion function. (This is why
    the OVRFisheye model needed to move to 6 radial terms: it has more
    distortion to correct for.)

    It's also relatively expensive to compute on since there is a trig
    function involved.

    
    """

    np_eps = np.finfo(np.float32).eps
    torch_eps = torch.tensor(np_eps, dtype=torch.float32)

    def stack(self, p, axis=0):
        assert len(p) > 0
        return torch.stack(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.stack(p, axis=axis)

    def concatenate(self, p, axis=0):
        return torch.cat(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.concatenate(p, axis=axis)

    def arctan2(self, x, y):
        return torch.atan2(x, y) if isinstance(x, torch.Tensor) else np.arctan2(x, y)

    def norm(self, x, ord=None, axis=None, keepdims=False):
        return torch.linalg.norm(x, ord=ord, dim=axis, keepdims=keepdims) if isinstance(x, torch.Tensor) else np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        
    def project(self, p):
        assert p.shape[-1] == 3
        lib = torch if isinstance(p, torch.Tensor) else np
        eps = self.torch_eps if lib is torch else self.np_eps            

        x, y, z = lib.moveaxis(p, -1, 0)
        r = lib.sqrt(x * x + y * y)
        s = self.arctan2(r, z) / lib.maximum(r, eps)
        return self.stack((x * s, y * s), axis=-1)

    def unproject(self, uv):
        assert uv.shape[-1] == 2
        lib = torch if isinstance(uv, torch.Tensor) else np            

        u, v = lib.moveaxis(uv, -1, 0)
        r = lib.sqrt(u * u + v * v)
        c = lib.cos(r)
        s = lib.sinc(r / np.pi)
        return self.stack([u * s, v * s, c], axis=-1)

    def project3(self, p):
        assert p.shape[-1] == 3
        lib = torch if isinstance(p, torch.Tensor) else np            
        eps = self.torch_eps if lib is torch else self.np_eps            

        xy = p[..., :2]
        z = p[..., 2]
        r2 = self.norm(xy, axis=-1)
        r3 = self.norm(p, axis=-1)
        s = self.arctan2(r2, z) / lib.maximum(r2, eps)

        return self.stack([xy[..., 0] * s, xy[..., 1] * s, r3 * lib.sign(z)], axis=-1)

    def unproject3(self, uvd):
        assert uvd.shape[-1] == 3
        lib = torch if isinstance(uvd, torch.Tensor) else np            

        u, v, d = lib.moveaxis(uvd, -1, 0)
        r = lib.sqrt(u * u + v * v)
        c = lib.cos(r)
        s = lib.sinc(r / np.pi)
        return lib.moveaxis(self.stack([u * s, v * s, c]) * d, 0, -1)



class PerspectiveProjection:
    np_eps = np.finfo(np.float32).eps
    torch_eps = torch.tensor(np_eps, dtype=torch.float32)

    def stack(self, p, axis=0):
        assert len(p) > 0
        return torch.stack(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.stack(p, axis=axis)

    def concatenate(self, p, axis=0):
        return torch.cat(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.concatenate(p, axis=axis)

    def norm(self, x, ord=None, axis=None, keepdims=False):
        return torch.linalg.norm(x, ord=ord, dim=axis, keepdims=keepdims) if isinstance(x, torch.Tensor) else np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        
    def rsqrt(self, x):
        return x ** -0.5

    def project(self, p):
        assert p.shape[-1] == 3
        lib = torch if isinstance(p, torch.Tensor) else np
        eps = self.torch_eps if lib is torch else self.np_eps            

        X, Y, Z = lib.moveaxis(p, -1, 0)

        Z_safe = Z + eps

        u = X / Z_safe
        v = Y / Z_safe

        # stack back into (...,2)
        return self.stack((u, v), axis=-1)

    def unproject(self, uv):
        assert uv.shape[-1] == 2
        lib = torch if isinstance(uv, torch.Tensor) else np            

        u, v = lib.moveaxis(uv, -1, 0)
        inv = self.rsqrt(u*u + v*v + 1.)
        d   = self.stack((u*inv, v*inv, inv), axis=-1)
        return d

