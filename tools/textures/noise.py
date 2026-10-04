"""Vectorised procedural noise for texture synthesis (numpy only).

noise3/fbm3 evaluate seamless 3D value noise at arbitrary world positions
(used with baked position maps, so patterns are continuous across UV seams).
tile_fbm2 produces seamlessly tiling 2D fractal noise via spectral synthesis.
"""
import numpy as np

M32 = np.uint64(0xFFFFFFFF)


def _hash(ix, iy, iz, seed):
    h = (ix.astype(np.uint64) * np.uint64(73856093)) ^ (iy.astype(np.uint64) * np.uint64(19349663)) \
        ^ (iz.astype(np.uint64) * np.uint64(83492791)) ^ np.uint64((seed * 2654435761) & 0xFFFFFFFF)
    h &= M32
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(1274126177)) & M32
    h ^= h >> np.uint64(16)
    return (h & np.uint64(0xFFFFFF)).astype(np.float32) / np.float32(16777216.0)


def noise3(P, freq=1.0, seed=0):
    """Value noise in [0,1] at points P (N,3)."""
    p = P * freq
    i = np.floor(p)
    f = (p - i).astype(np.float32)
    i = i.astype(np.int64) + 1_000_000
    u = f * f * (3 - 2 * f)
    out = np.zeros(len(P), np.float32)
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1 - u[:, 2]
                out += wx * wy * wz * _hash(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def fbm3(P, freq=1.0, octaves=4, seed=0, gain=0.5, lac=2.03):
    amp, tot, s = 1.0, 0.0, np.zeros(len(P), np.float32)
    for o in range(octaves):
        s += amp * noise3(P, freq * lac ** o, seed + 17 * o)
        tot += amp
        amp *= gain
    return s / tot


def tile_fbm2(res, base_freq=4, octaves=6, seed=0, gain=0.55, aniso=(1.0, 1.0)):
    """Seamless 2D fractal noise in [0,1] (spectral synthesis with a 1/f^beta falloff)."""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(res) * res / aniso[0]
    fy = np.fft.fftfreq(res) * res / aniso[1]
    FX, FY = np.meshgrid(fx, fy)
    r = np.sqrt(FX ** 2 + FY ** 2)
    beta = -np.log2(gain) + 1.0
    amp = np.where(r > 0, (np.maximum(r, base_freq * 0.5) / base_freq) ** (-beta), 0.0)
    amp[r > base_freq * 2 ** octaves] = 0
    phase = rng.uniform(0, 2 * np.pi, (res, res))
    spec = amp * np.exp(1j * phase)
    img = np.real(np.fft.ifft2(spec))
    img -= img.min()
    img /= img.max() + 1e-9
    return img.astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def height_to_normal(h, strength=1.0, wrap=True):
    """OpenGL (Y+) tangent-space normal map from a height field (rows = +V going down in image space)."""
    if wrap:
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    else:
        dx = np.gradient(h, axis=1)
        dy = np.gradient(h, axis=0)
    # image row index increases downwards (= -V); OpenGL green = +V up
    nx = -dx * strength
    ny = dy * strength
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def scratches(res, count, length=(0.02, 0.15), width=1, seed=0, angle=None, wrap=True):
    """Seamless scratch mask (float 0..1) drawn with Pillow."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    img = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(img)
    for _ in range(count):
        x, y = rng.uniform(0, res, 2)
        L = rng.uniform(*length) * res
        a = rng.uniform(0, np.pi) if angle is None else angle + rng.normal(0, 0.25)
        x2, y2 = x + L * np.cos(a), y + L * np.sin(a)
        val = int(rng.uniform(90, 255))
        for ox in ((-res, 0, res) if wrap else (0,)):
            for oy in ((-res, 0, res) if wrap else (0,)):
                d.line([(x + ox, y + oy), (x2 + ox, y2 + oy)], fill=val, width=width)
    return np.asarray(img, np.float32) / 255.0
