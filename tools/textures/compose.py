"""Compose HD PBR textures for the FD-25 from the baked helper maps + procedural noise,
and write main.materials.json.

Run:  python -m tools.textures.compose
"""
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from tools.spec import machine as M
from tools import kinematics as K
from tools.textures.noise import fbm3, noise3, tile_fbm2, smoothstep, height_to_normal, scratches

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build")
VDIR = os.path.join(ROOT, "mod", "vehicles", M.MODEL_ID)
TEX = os.path.join(VDIR, "textures")
GAME_TEX = f"/vehicles/{M.MODEL_ID}/textures"
PREVIEW = {}

HI = 4096
LO = 2048


# ----------------------------------------------------------------------------- io helpers
def load_bake(atlas, res):
    d = np.load(os.path.join(BUILD, "bake", f"{atlas}.npz"))
    src = d["ao"].shape[0]
    z = res / src
    out = {}
    for k in ("ao", "edge", "point"):
        out[k] = ndimage.zoom(d[k], z, order=1) if z != 1 else d[k].copy()
    for k in ("pos", "nrm"):
        out[k] = np.stack([ndimage.zoom(d[k][..., i], z, order=1) if z != 1 else d[k][..., i] for i in range(3)], -1)
    out["cov"] = (ndimage.zoom(d["cov"], z, order=0) if z != 1 else d["cov"]) > 0.5
    # Blender image rows start at the bottom; flip to image convention (row 0 = top = v 1)
    for k in out:
        out[k] = np.flipud(out[k])
    out["res"] = res
    return out


def dilate(arr, cov, iterations=None):
    """Fill uncovered texels with the nearest covered value (UV padding for mip-mapping)."""
    if cov.all():
        return arr
    idx = ndimage.distance_transform_edt(~cov, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def save(name, arr, kind):
    """kind: color (sRGB RGB/RGBA), data (linear L), normal (RGB)."""
    os.makedirs(TEX, exist_ok=True)
    path = os.path.join(TEX, name)
    a = arr
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if kind == "data" and a.ndim == 3:
        a = a[..., 0]
    mode = "L" if a.ndim == 2 else {3: "RGB", 4: "RGBA"}[a.shape[2]]
    Image.fromarray(a, mode).save(path, optimize=True)
    return f"{GAME_TEX}/{name}", path


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def lerp(a, b, t):
    t = t[..., None] if (np.ndim(t) == np.ndim(a) - 1 and np.ndim(a) == 3) else t
    return a + (b - a) * t


def field(B, fn):
    """Evaluate a function of (pos, nrm) on covered texels only -> full image (0 elsewhere)."""
    cov = B["cov"]
    out = np.zeros(cov.shape, np.float32)
    P = B["pos"][cov]
    out[cov] = fn(P, B["nrm"][cov])
    return out


def rgb(c):
    return np.array(c, np.float32)


# ----------------------------------------------------------------------------- masks shared by painted atlases
def wear_masks(B, seed, strength=1.0):
    edge = np.clip(B["edge"] * 9.0, 0, 1)
    convex = edge * smoothstep(0.49, 0.56, B["point"]) + edge * 0.35
    chips = field(B, lambda P, N: fbm3(P, 16.0, 4, seed))
    clusters = field(B, lambda P, N: fbm3(P, 2.2, 3, seed + 5))
    speck = field(B, lambda P, N: fbm3(P, 55.0, 2, seed + 9))
    edge_wear = np.clip(convex * 1.4, 0, 1) * smoothstep(0.40, 0.62, chips + 0.30 * convex * strength)
    scattered = smoothstep(0.78, 0.86, speck) * smoothstep(0.45, 0.75, clusters) * 0.9 * strength
    wear = np.clip(edge_wear + scattered, 0, 1)
    return wear, chips, clusters


def dirt_masks(B, seed, mud_height=None, dust=1.0):
    nz = B["nrm"][..., 2]
    ao = np.clip(B["ao"], 0, 1)
    n1 = field(B, lambda P, N: fbm3(P, 3.0, 4, seed))
    streak = field(B, lambda P, N: fbm3(P * np.array([9.0, 9.0, 0.9]), 1.0, 4, seed + 3))
    up = smoothstep(0.35, 0.9, nz) * (0.45 + 0.6 * n1) * dust
    cav = np.power(1 - ao, 1.4) * (0.5 + 0.6 * n1)
    streaks = smoothstep(0.55, 0.75, streak) * (1 - smoothstep(0.5, 0.8, np.abs(nz))) * 0.55
    mud = 0
    if mud_height is not None:
        z = B["pos"][..., 2]
        mud = smoothstep(mud_height, mud_height * 0.35, z) * (0.55 + 0.6 * field(B, lambda P, N: fbm3(P, 6.0, 4, seed + 7)))
    dirt = np.clip(up * 0.45 + cav * 0.50 + streaks * 0.8 + mud, 0, 0.88)
    return dirt, n1


# ----------------------------------------------------------------------------- painted atlases (3 stages)
def paint_atlas(atlas, seed, mud_height=None, wear_strength=1.0, dust=1.0):
    t0 = time.time()
    Bh = load_bake(atlas, HI)
    Bl = load_bake(atlas, LO)
    wear, chips, clusters = wear_masks(Bh, seed, wear_strength)
    wear_lo = ndimage.zoom(wear, LO / HI, order=1)
    # --- paint opacity (4K, crisp chips)
    paint_o = 1 - wear
    paint_o = dilate(paint_o, Bh["cov"])
    files = {}
    files["paint_o"] = save(f"fd25_{atlas}_paint_o.data.png", paint_o, "data")
    # --- under layer (primer / bare steel / rust) at 2K
    rust_n = field(Bl, lambda P, N: fbm3(P, 5.0, 4, seed + 11))
    steel_n = field(Bl, lambda P, N: fbm3(P, 30.0, 3, seed + 12))
    core = smoothstep(0.45, 0.9, wear_lo)
    rust = np.clip(smoothstep(0.55, 0.72, rust_n) * (0.4 + 0.6 * (1 - np.clip(Bl["ao"], 0, 1))) + 0.15 * core, 0, 1)
    primer = rgb((0.30, 0.15, 0.10))
    steel = rgb((0.43, 0.43, 0.42)) * (0.85 + 0.3 * steel_n)[..., None]
    rustc = rgb((0.26, 0.12, 0.05)) * (0.8 + 0.4 * rust_n)[..., None]
    under = lerp(np.broadcast_to(primer, steel.shape), steel, core)
    under = lerp(under, rustc, rust * 0.8)
    under = dilate(under, Bl["cov"])
    files["under"] = save(f"fd25_{atlas}_under.color.png", to_srgb(under), "color")
    metal = np.clip(core * 0.85 * (1 - rust), 0, 1)
    files["metal"] = save(f"fd25_{atlas}_m.data.png", dilate(ndimage.zoom(metal, 0.5, order=1),
                                                              ndimage.zoom(Bl["cov"], 0.5, order=0)), "data")
    rough_u = np.clip(0.42 + 0.12 * steel_n + 0.45 * rust + 0.25 * (1 - core), 0, 1)
    files["rough"] = save(f"fd25_{atlas}_r.data.png", dilate(rough_u, Bl["cov"]), "data")
    # --- paint tint (multiplied by the in-game colour), subtle fading
    fade = field(Bl, lambda P, N: fbm3(P, 1.2, 3, seed + 21))
    tint = np.clip(0.86 + 0.10 * fade + 0.03 * smoothstep(0.6, 1.0, Bl["nrm"][..., 2]), 0, 1)
    tint_img = np.repeat(tint[..., None], 3, -1) * rgb((1.0, 0.985, 0.96))
    tint_img = dilate(ndimage.zoom(tint_img, (0.5, 0.5, 1), order=1), ndimage.zoom(Bl["cov"], 0.5, order=0))
    files["paint"] = save(f"fd25_{atlas}_paint.color.png", to_srgb(tint_img), "color")
    prough = np.clip(0.36 + 0.18 * fade + 0.25 * smoothstep(0.5, 1.0, Bl["nrm"][..., 2]) * fade, 0, 1)
    files["paint_r"] = save(f"fd25_{atlas}_paint_r.data.png",
                            dilate(ndimage.zoom(prough, 0.5, order=1), ndimage.zoom(Bl["cov"], 0.5, order=0)), "data")
    # --- dirt layer
    dirt, n1 = dirt_masks(Bl, seed + 30, mud_height, dust)
    dirt_c = lerp(np.broadcast_to(rgb((0.42, 0.37, 0.30)), (LO, LO, 3)), rgb((0.16, 0.13, 0.10)),
                  np.clip(np.power(1 - np.clip(Bl["ao"], 0, 1), 1.2) * 1.3, 0, 1))
    dirt_c = dirt_c * (0.85 + 0.3 * n1)[..., None]
    files["dirt"] = save(f"fd25_{atlas}_dirt.color.png", to_srgb(dilate(dirt_c, Bl["cov"])), "color")
    files["dirt_o"] = save(f"fd25_{atlas}_dirt_o.data.png", dilate(dirt, Bl["cov"]), "data")
    # --- AO and normal
    ao = np.clip(0.25 + 0.75 * np.clip(Bl["ao"], 0, 1), 0, 1)
    files["ao"] = save(f"fd25_{atlas}_ao.data.png", dilate(ao, Bl["cov"]), "data")
    paint_step = ndimage.gaussian_filter(1 - wear, 1.2)
    pit = field(Bh, lambda P, N: fbm3(P, 90.0, 2, seed + 40))
    dent = field(Bh, lambda P, N: fbm3(P, 1.5, 3, seed + 41))
    h = paint_step * 0.9 + (1 - paint_step) * pit * 1.2 + dent * 2.0
    h = ndimage.zoom(h, LO / HI, order=1)
    nm = height_to_normal(dilate(h, Bl["cov"]), strength=6.0, wrap=False)
    files["normal"] = save(f"fd25_{atlas}_nm.normal.png", nm, "normal")
    # --- preview composite (default paint) for Blender renders
    paint_col = rgb(M.PAINTS[M.DEFAULT_PAINT])
    lin_paint = np.power(paint_col, 2.2) * tint[..., None]
    po = ndimage.zoom(paint_o, LO / HI, order=1)
    comp = lerp(under, lin_paint, po)
    comp = lerp(comp, dirt_c, dirt)
    comp *= (0.35 + 0.65 * np.clip(Bl["ao"], 0, 1))[..., None]
    pv = save_preview(f"{atlas}_base", to_srgb(dilate(comp, Bl["cov"])))
    pr = save_preview(f"{atlas}_rough", np.clip(lerp(rough_u, prough, po) * (1 - dirt) + 0.93 * dirt, 0, 1))
    pm = save_preview(f"{atlas}_metal", np.clip(metal * (1 - po) * (1 - dirt), 0, 1))
    PREVIEW_MAT[atlas] = dict(base=pv, rough=pr, metal=pm, normal=files["normal"][1])
    print(f"  {atlas}: painted atlas in {time.time() - t0:.0f}s")
    return files


PREVIEW_MAT = {}


def save_preview(name, arr):
    os.makedirs(os.path.join(BUILD, "preview_tex"), exist_ok=True)
    path = os.path.join(BUILD, "preview_tex", f"{name}.png")
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(a).save(path)
    return path


# ----------------------------------------------------------------------------- single-stage atlases
def ring_mask(B):
    """Blue rotator ring: texels whose position lies on the ring band around the attachment axis."""
    W = K.solve()["W"]
    o = W("head", 0.0, 0.0, 0.0)
    a_dir = W("head", 1.0, 0.0, 0.0) - o
    a_dir /= np.linalg.norm(a_dir)
    P = B["pos"] - o
    a = P @ a_dir
    radial = np.linalg.norm(P - a[..., None] * a_dir, axis=-1)
    # the slewing ring itself plus the lower band of the rotator housing (the tall blue band in the reel)
    a0 = M.ATT.ring_a[0] - 0.09
    return (B["cov"] & (a > a0) & (a < M.ATT.ring_a[1] + 0.005) & (radial > 0.20)).astype(np.float32)


def solid_atlas(atlas, base_col, seed, res_base, wear_strength=1.4, mud_height=None, dust=1.0, extra=None):
    t0 = time.time()
    Bh = load_bake(atlas, res_base)
    Bl = load_bake(atlas, min(LO, res_base))
    wear, chips, clusters = wear_masks(Bh, seed, wear_strength)
    paint = np.broadcast_to(rgb(base_col), (res_base, res_base, 3)).copy()
    var = field(Bh, lambda P, N: fbm3(P, 3.0, 4, seed + 3))
    paint *= (0.85 + 0.3 * var)[..., None]
    ring = None
    if extra == "ring":
        ring = ring_mask(Bh)
        paint = lerp(paint, rgb(M.ATT.ring_color) * (0.85 + 0.3 * var)[..., None], ring)
    steel_n = field(Bh, lambda P, N: fbm3(P, 30.0, 3, seed + 12))
    rust_n = field(Bh, lambda P, N: fbm3(P, 5.0, 4, seed + 11))
    steel = rgb((0.46, 0.46, 0.45)) * (0.85 + 0.3 * steel_n)[..., None]
    rust = smoothstep(0.58, 0.75, rust_n) * (0.5 + 0.5 * (1 - np.clip(Bh["ao"], 0, 1)))
    rustc = rgb((0.24, 0.11, 0.05))
    base = lerp(paint, steel, wear)
    base = lerp(base, rustc, rust * 0.7)
    dirt, n1 = dirt_masks(Bh, seed + 30, mud_height, dust)
    dirt_c = lerp(np.broadcast_to(rgb((0.27, 0.24, 0.20)), base.shape), rgb((0.10, 0.085, 0.07)),
                  np.clip(np.power(1 - np.clip(Bh["ao"], 0, 1), 1.2) * 1.3, 0, 1)) * (0.85 + 0.3 * n1)[..., None]
    base = lerp(base, dirt_c, dirt * 0.6)
    files = {}
    files["base"] = save(f"fd25_{atlas}_b.color.png", to_srgb(dilate(base, Bh["cov"])), "color")
    scale = Bl["res"] / res_base
    z = (lambda a: ndimage.zoom(a, scale, order=1)) if scale != 1 else (lambda a: a)
    metal = np.clip(wear * 0.9 * (1 - rust) * (1 - 0.6 * dirt), 0, 1)
    rough = np.clip(0.48 + 0.15 * var - 0.2 * wear + 0.35 * rust + 0.3 * dirt, 0.05, 1)
    if ring is not None:
        rough = rough - 0.12 * ring * (1 - dirt)
    files["metal"] = save(f"fd25_{atlas}_m.data.png", dilate(z(metal), Bl["cov"]), "data")
    files["rough"] = save(f"fd25_{atlas}_r.data.png", dilate(z(rough), Bl["cov"]), "data")
    ao = np.clip(0.25 + 0.75 * np.clip(Bl["ao"], 0, 1), 0, 1)
    files["ao"] = save(f"fd25_{atlas}_ao.data.png", dilate(ao, Bl["cov"]), "data")
    pit = field(Bh, lambda P, N: fbm3(P, 90.0, 2, seed + 40))
    dent = field(Bh, lambda P, N: fbm3(P, 1.5, 3, seed + 41))
    h = ndimage.gaussian_filter(1 - wear, 1.0) * 0.7 + rust * pit * 1.5 + dent * 2.0
    nm = height_to_normal(dilate(z(h), Bl["cov"]), strength=6.0, wrap=False)
    files["normal"] = save(f"fd25_{atlas}_nm.normal.png", nm, "normal")
    PREVIEW_MAT[atlas] = dict(base=files["base"][1], rough=files["rough"][1], metal=files["metal"][1],
                              normal=files["normal"][1])
    print(f"  {atlas}: solid atlas in {time.time() - t0:.0f}s")
    return files


def interior_atlas():
    t0 = time.time()
    res = 1024
    B = load_bake("int", res)
    z = B["pos"][..., 2]
    x = B["pos"][..., 0]
    y = B["pos"][..., 1]
    zf = M.CAB.z0 + 0.18
    floor = (z < zf + 0.03)
    seat = (np.abs(x - 0.92) < 0.27) & (y > -0.30) & (y < 0.34) & (z > zf + 0.25) & (z < zf + 1.20)
    liner = z > M.CAB.z1 - 0.15
    weave = tile_fbm2(res, 64, 3, seed=5)
    n = field(B, lambda P, N: fbm3(P, 8.0, 3, 77))
    base = np.broadcast_to(rgb((0.07, 0.072, 0.075)), (res, res, 3)).copy()
    base = np.where(floor[..., None], rgb((0.035, 0.035, 0.035)), base)
    base = np.where(seat[..., None], rgb((0.10, 0.10, 0.11)) * (0.8 + 0.4 * weave)[..., None], base)
    base = np.where(liner[..., None], rgb((0.26, 0.26, 0.25)), base)
    base *= (0.85 + 0.3 * n)[..., None]
    rough = np.where(seat, 0.9, np.where(floor, 0.85, 0.55)) + 0.08 * n
    files = {"base": save("fd25_int_b.color.png", to_srgb(dilate(base, B["cov"])), "color"),
             "rough": save("fd25_int_r.data.png", dilate(np.clip(rough, 0, 1), B["cov"]), "data"),
             "ao": save("fd25_int_ao.data.png", dilate(np.clip(0.3 + 0.7 * B["ao"], 0, 1), B["cov"]), "data")}
    h = np.where(seat, weave, 0) + n * 0.5
    files["normal"] = save("fd25_int_nm.normal.png", height_to_normal(dilate(h, B["cov"]), 3.0, wrap=False), "normal")
    PREVIEW_MAT["int"] = dict(base=files["base"][1], rough=files["rough"][1], normal=files["normal"][1])
    print(f"  int: interior atlas in {time.time() - t0:.0f}s")
    return files


# ----------------------------------------------------------------------------- tiling materials
def tiling_textures():
    out = {}
    R = 1024
    # worn steel
    n = tile_fbm2(R, 6, 6, seed=1)
    sc = scratches(R, 900, (0.02, 0.12), 1, seed=2)
    base = rgb((0.40, 0.40, 0.39)) * (0.8 + 0.35 * n)[..., None] + sc[..., None] * 0.15
    rust = smoothstep(0.62, 0.8, tile_fbm2(R, 3, 5, seed=3))
    base = lerp(base, rgb((0.25, 0.12, 0.05)), rust * 0.6)
    out["steel"] = dict(
        base=save("fd25_steel_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_steel_r.data.png", np.clip(0.38 + 0.15 * n - 0.15 * sc + 0.45 * rust, 0, 1), "data"),
        metal=save("fd25_steel_m.data.png", np.clip(0.9 - 0.8 * rust, 0, 1), "data"),
        normal=save("fd25_steel_nm.normal.png", height_to_normal(n * 0.6 + sc * 0.4 + rust * tile_fbm2(R, 40, 2, 4), 3.0), "normal"))
    # dark painted steel (sprockets, idlers)
    wear = smoothstep(0.66, 0.78, tile_fbm2(R, 10, 5, seed=5))
    base = lerp(np.broadcast_to(rgb((0.05, 0.05, 0.05)), (R, R, 3)) * (0.8 + 0.4 * n)[..., None], rgb((0.38, 0.37, 0.35)), wear)
    mud = smoothstep(0.5, 0.8, tile_fbm2(R, 4, 5, seed=6))
    base = lerp(base, rgb((0.30, 0.25, 0.19)), mud * 0.7)
    out["steel_dark"] = dict(
        base=save("fd25_steel_dark_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_steel_dark_r.data.png", np.clip(0.55 - 0.2 * wear + 0.35 * mud, 0, 1), "data"),
        metal=save("fd25_steel_dark_m.data.png", np.clip(wear * 0.85, 0, 1), "data"),
        normal=save("fd25_steel_dark_nm.normal.png", height_to_normal(n * 0.5 + mud * 0.6, 2.5), "normal"))
    # chrome rods (vertical streaks along the rod axis = local X -> box-projected U)
    streak = tile_fbm2(R, 2, 6, seed=7, aniso=(0.05, 1.0))
    grime = smoothstep(0.6, 0.85, tile_fbm2(R, 5, 5, seed=8))
    base = lerp(np.broadcast_to(rgb((0.80, 0.81, 0.83)), (R, R, 3)), rgb((0.25, 0.22, 0.18)), grime * 0.6)
    out["chrome"] = dict(
        base=save("fd25_chrome_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_chrome_r.data.png", np.clip(0.06 + 0.06 * streak + 0.5 * grime, 0, 1), "data"),
        normal=save("fd25_chrome_nm.normal.png", height_to_normal(streak * 0.3, 1.0), "normal"))
    # rubber hoses
    fine = tile_fbm2(R, 64, 3, seed=9)
    dust = smoothstep(0.55, 0.85, tile_fbm2(R, 4, 5, seed=10))
    base = lerp(np.broadcast_to(rgb((0.025, 0.025, 0.025)), (R, R, 3)) * (0.8 + 0.4 * fine)[..., None], rgb((0.30, 0.27, 0.23)), dust * 0.5)
    out["rubber"] = dict(
        base=save("fd25_rubber_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_rubber_r.data.png", np.clip(0.72 + 0.15 * fine + 0.1 * dust, 0, 1), "data"),
        normal=save("fd25_rubber_nm.normal.png", height_to_normal(fine * 0.4, 2.0), "normal"))
    # black paint / plastic
    base = lerp(np.broadcast_to(rgb((0.035, 0.035, 0.037)), (R, R, 3)) * (0.85 + 0.3 * n)[..., None], rgb((0.33, 0.30, 0.25)), dust * 0.55)
    sc2 = scratches(R, 300, (0.01, 0.05), 1, seed=11)
    base = base + sc2[..., None] * 0.12
    out["black"] = dict(
        base=save("fd25_black_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_black_r.data.png", np.clip(0.42 + 0.2 * n + 0.35 * dust, 0, 1), "data"),
        normal=save("fd25_black_nm.normal.png", height_to_normal(n * 0.3 + sc2 * 0.3, 2.0), "normal"))
    # diamond tread plate: elongated lozenges, alternating +-45 degrees in a checkerboard
    Rg = 1024
    cells = 8
    yy, xx = (np.mgrid[0:Rg, 0:Rg] + 0.5) / Rg * cells
    cx, cy = np.floor(xx), np.floor(yy)
    u, v = xx - cx - 0.5, yy - cy - 0.5
    sgn_ = np.where((cx + cy) % 2 == 0, 1.0, -1.0)
    ru = (u + sgn_ * v) / math.sqrt(2)
    rv = (-sgn_ * u + v) / math.sqrt(2)
    lozenge = np.abs(ru) / 0.36 + np.abs(rv) / 0.085
    hgt = smoothstep(1.05, 0.80, lozenge)
    worn = hgt * smoothstep(0.40, 0.62, tile_fbm2(Rg, 6, 4, seed=12))
    gdust = smoothstep(0.45, 0.85, tile_fbm2(Rg, 3, 5, seed=13)) * (1 - hgt * 0.6)
    base = lerp(np.broadcast_to(rgb((0.12, 0.12, 0.12)), (Rg, Rg, 3)) * (0.85 + 0.3 * tile_fbm2(Rg, 8, 4, 14))[..., None],
                rgb((0.48, 0.48, 0.47)), worn)
    base = lerp(base, rgb((0.33, 0.30, 0.25)), gdust * 0.7)
    out["grating"] = dict(
        base=save("fd25_grating_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_grating_r.data.png", np.clip(0.62 - 0.32 * worn + 0.25 * gdust, 0, 1), "data"),
        metal=save("fd25_grating_m.data.png", np.clip(0.25 + 0.65 * worn - 0.3 * gdust, 0, 1), "data"),
        normal=save("fd25_grating_nm.normal.png", height_to_normal(ndimage.gaussian_filter(hgt, 1.0), 6.0), "normal"))
    # track shoe steel (rusty, polished on grousers)
    Rt = 2048
    nt = tile_fbm2(Rt, 6, 6, seed=14)
    rustt = smoothstep(0.42, 0.7, tile_fbm2(Rt, 4, 6, seed=15))
    mudt = smoothstep(0.55, 0.8, tile_fbm2(Rt, 3, 6, seed=16))
    sct = scratches(Rt, 1500, (0.01, 0.06), 2, seed=17, angle=math.pi / 2)
    base = lerp(np.broadcast_to(rgb((0.16, 0.155, 0.15)), (Rt, Rt, 3)) * (0.75 + 0.5 * nt)[..., None], rgb((0.20, 0.11, 0.06)), rustt * 0.55)
    base = lerp(base, rgb((0.28, 0.25, 0.20)), mudt * 0.6) + sct[..., None] * 0.18
    out["track"] = dict(
        base=save("fd25_track_b.color.png", to_srgb(base), "color"),
        rough=save("fd25_track_r.data.png", np.clip(0.6 + 0.3 * rustt - 0.3 * sct + 0.2 * mudt, 0, 1), "data"),
        metal=save("fd25_track_m.data.png", np.clip(0.55 - 0.45 * rustt - 0.4 * mudt + 0.4 * sct, 0, 1), "data"),
        normal=save("fd25_track_nm.normal.png", height_to_normal(nt * 0.6 + rustt * tile_fbm2(Rt, 60, 2, 18) * 0.8 + mudt * 0.5, 4.0), "normal"))
    # cylinder barrel paint (tint, multiplied by vehicle colour) + grime overlay
    tint = 0.88 + 0.1 * tile_fbm2(R, 2, 4, seed=19)
    out["cyl"] = dict(
        base=save("fd25_cyl_paint_b.color.png", to_srgb(np.repeat(tint[..., None], 3, -1)), "color"),
        grime=save("fd25_cyl_grime_o.data.png", np.clip(smoothstep(0.5, 0.85, tile_fbm2(R, 4, 6, seed=20)) * 0.8, 0, 1), "data"),
        rough=save("fd25_cyl_paint_r.data.png", np.clip(0.38 + 0.2 * tile_fbm2(R, 6, 4, seed=21), 0, 1), "data"))
    # glass dirt (opacity) and lens patterns
    gd = smoothstep(0.45, 0.9, tile_fbm2(R, 3, 6, seed=22))
    out["glass"] = dict(opacity=save("fd25_glass_o.data.png", np.clip(0.18 + 0.55 * gd, 0, 1), "data"),
                        base=save("fd25_glass_b.color.png", to_srgb(lerp(np.broadcast_to(rgb((0.55, 0.62, 0.62)), (R, R, 3)),
                                                                         rgb((0.42, 0.37, 0.30)), gd)), "color"))
    Rl = 512
    yy, xx = np.mgrid[0:Rl, 0:Rl] / Rl
    prisms = (np.sin(xx * 2 * math.pi * 16) * np.sin(yy * 2 * math.pi * 16)) * 0.5 + 0.5
    out["lamp"] = dict(normal=save("fd25_lamp_nm.normal.png", height_to_normal(prisms, 2.5), "normal"),
                       glow=save("fd25_lamp_g.color.png", np.repeat((0.75 + 0.25 * prisms)[..., None], 3, -1) * rgb((1.0, 0.97, 0.90)), "color"))
    # detail normals for UV1 (orange peel + micro scratches), world-scale
    peel = tile_fbm2(R, 48, 3, seed=23)
    msc = scratches(R, 2000, (0.005, 0.03), 1, seed=24)
    out["detail"] = dict(normal=save("fd25_detail_paint_nm.normal.png", height_to_normal(peel * 0.25 + msc * 0.35, 1.5), "normal"))
    return out


# ----------------------------------------------------------------------------- decals
def font(size, bold=True):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def decal_atlas():
    R = 2048
    img = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def band(v0, v1):  # uv v -> image rows (v up)
        return int((1 - v1) * R), int((1 - v0) * R)
    # brand (v 0.75..1.0)
    y0, y1 = band(0.75, 1.0)
    f = font(int((y1 - y0) * 0.62))
    txt = "FERROVORE"
    w = d.textlength(txt, font=f)
    d.text(((R - w) / 2, y0 + (y1 - y0) * 0.12), txt, font=f, fill=(18, 18, 18, 255))
    # hazard stripes (v 0.50..0.60)
    y0, y1 = band(0.50, 0.60)
    d.rectangle([0, y0, R, y1], fill=(235, 170, 20, 255))
    step = (y1 - y0)
    for k in range(-2, R // step + 3):
        x = k * step
        d.polygon([(x, y1), (x + step * 0.5, y1), (x + step * 1.5, y0), (x + step, y0)], fill=(20, 20, 20, 255))
    # model badges (v 0.62..0.74): left = "FD-25", right = "FD-25 DISMANTLER"
    y0, y1 = band(0.62, 0.74)
    f2 = font(int((y1 - y0) * 0.70))
    d.text((R * 0.02, y0 + (y1 - y0) * 0.1), "FD-25", font=f2, fill=(20, 20, 20, 255))
    f3 = font(int((y1 - y0) * 0.42))
    d.text((R * 0.52, y0 + (y1 - y0) * 0.05), "FD-25", font=f2, fill=(20, 20, 20, 255))
    d.text((R * 0.80, y0 + (y1 - y0) * 0.30), "ELV", font=f3, fill=(20, 20, 20, 255))
    # warning labels (v 0.30..0.46): "KEEP CLEAR - SWING AREA" style
    y0, y1 = band(0.30, 0.46)
    d.rectangle([R * 0.02, y0, R * 0.48, y1], fill=(250, 210, 30, 255), outline=(10, 10, 10, 255), width=8)
    f4 = font(int((y1 - y0) * 0.25))
    d.text((R * 0.05, y0 + 20), "WARNING", font=f4, fill=(10, 10, 10, 255))
    d.text((R * 0.05, y0 + (y1 - y0) * 0.5), "KEEP CLEAR OF SWING AREA", font=font(int((y1 - y0) * 0.16)), fill=(10, 10, 10, 255))
    arr = np.asarray(img).astype(np.float32) / 255
    # light weathering on decals
    wear = smoothstep(0.62, 0.8, tile_fbm2(R, 12, 5, seed=30))
    arr[..., 3] *= 1 - wear * 0.8
    rgb_ = arr[..., :3]
    a = arr[..., 3]
    base = save("fd25_decals_b.color.png", np.dstack([to_srgb(np.power(rgb_, 2.2)), np.ones((R, R))]).astype(np.float32), "color")
    opac = save("fd25_decals_o.data.png", a, "data")
    PREVIEW_MAT["decals"] = dict(base=base[1], alpha=opac[1])
    return dict(base=base, opacity=opac)


# ----------------------------------------------------------------------------- materials.json
def stage(**kw):
    return {k: v for k, v in kw.items() if v is not None}


def mat(name, stages, layers=None, **extra):
    st = stages + [{}] * (4 - len(stages))
    d = {"name": name, "mapTo": name, "class": "Material", "Stages": st,
         "dynamicCubemap": True, "materialTag0": "beamng", "materialTag1": "vehicle", "version": 1.5}
    if layers:
        d["activeLayers"] = layers
    d.update(extra)
    return d


def write_materials(P, T, D):
    G = lambda f: f[0]  # game path
    det = G(T["detail"]["normal"])
    mats = {}
    for atlas, mname in (("upper", "fd25_paint_upper"), ("arm", "fd25_paint_arm")):
        f = P[atlas]
        common = dict(normalMap=G(f["normal"]), ambientOcclusionMap=G(f["ao"]))
        mats[mname] = mat(mname, [
            stage(baseColorMap=G(f["under"]), metallicMap=G(f["metal"]), roughnessMap=G(f["rough"]), **common),
            stage(baseColorMap=G(f["paint"]), instanceBaseColor=True, opacityMap=G(f["paint_o"]),
                  roughnessMap=G(f["paint_r"]), metallicFactor=0, clearCoatFactor=0.35, clearCoatRoughnessFactor=0.35,
                  detailNormalMap=det, detailNormalMapStrength=0.35, detailScale=[3, 3], normalDetailMapUseUV=1,
                  **common),
            stage(baseColorMap=G(f["dirt"]), opacityMap=G(f["dirt_o"]), roughnessFactor=0.93, metallicFactor=0, **common),
        ], layers=3)
    for atlas, mname in (("att", "fd25_attachment"), ("uc", "fd25_undercarriage")):
        f = P[atlas]
        mats[mname] = mat(mname, [stage(baseColorMap=G(f["base"]), metallicMap=G(f["metal"]), roughnessMap=G(f["rough"]),
                                        normalMap=G(f["normal"]), ambientOcclusionMap=G(f["ao"]),
                                        detailNormalMap=det, detailNormalMapStrength=0.3, detailScale=[3, 3],
                                        normalDetailMapUseUV=1)])
    f = P["int"]
    mats["fd25_interior"] = mat("fd25_interior", [stage(baseColorMap=G(f["base"]), roughnessMap=G(f["rough"]),
                                                        normalMap=G(f["normal"]), ambientOcclusionMap=G(f["ao"]), metallicFactor=0)])
    for key, mname in (("steel", "fd25_steel"), ("steel_dark", "fd25_steel_dark"), ("track", "fd25_track"),
                       ("grating", "fd25_grating")):
        t = T[key]
        mats[mname] = mat(mname, [stage(baseColorMap=G(t["base"]), roughnessMap=G(t["rough"]), metallicMap=G(t["metal"]),
                                        normalMap=G(t["normal"]))])
    for key, mname, met in (("chrome", "fd25_chrome", 1.0), ("rubber", "fd25_rubber", 0.0), ("black", "fd25_black", 0.0)):
        t = T[key]
        mats[mname] = mat(mname, [stage(baseColorMap=G(t["base"]), roughnessMap=G(t["rough"]), metallicFactor=met,
                                        normalMap=G(t["normal"]))])
    t = T["cyl"]
    mats["fd25_cyl_paint"] = mat("fd25_cyl_paint", [
        stage(baseColorMap=G(t["base"]), instanceBaseColor=True, roughnessMap=G(t["rough"]), metallicFactor=0,
              clearCoatFactor=0.3),
        stage(baseColorFactor=[0.30, 0.26, 0.21, 1], opacityMap=G(t["grime"]), roughnessFactor=0.9, metallicFactor=0),
    ], layers=2)
    g = T["glass"]
    for mname, op in (("fd25_glass", None), ("fd25_glass_int", 0.08)):
        st = stage(baseColorMap=G(g["base"]), metallicFactor=0.6, roughnessFactor=0.05,
                   opacityMap=G(g["opacity"]) if op is None else None, opacityFactor=op)
        mats[mname] = mat(mname, [st], castShadows=False, translucent=True, translucentBlendOp="PreMulAlpha",
                          translucentRecvShadows=True)
    lamp = T["lamp"]
    mats["fd25_lamp"] = mat("fd25_lamp", [stage(baseColorFactor=[0.82, 0.82, 0.80, 1], roughnessFactor=0.08,
                                                metallicFactor=0.3, normalMap=G(lamp["normal"]))])
    mats["fd25_lamp_on"] = mat("fd25_lamp_on", [stage(baseColorFactor=[0.95, 0.95, 0.92, 1], roughnessFactor=0.08,
                                                      metallicFactor=0.3, normalMap=G(lamp["normal"]),
                                                      emissiveMap=G(lamp["glow"]), emissiveFactor=[1, 1, 1],
                                                      emissiveIntensityNits=9000)])
    mats["fd25_beacon"] = mat("fd25_beacon", [stage(baseColorFactor=[0.95, 0.42, 0.03, 1], roughnessFactor=0.15,
                                                    metallicFactor=0, opacityFactor=0.85)],
                              translucent=True, translucentBlendOp="PreMulAlpha")
    mats["fd25_beacon_on"] = mat("fd25_beacon_on", [stage(baseColorFactor=[1.0, 0.55, 0.05, 1], roughnessFactor=0.15,
                                                          metallicFactor=0, emissiveFactor=[1.0, 0.5, 0.05],
                                                          emissiveIntensityNits=12000)])
    mats["fd25_decals"] = mat("fd25_decals", [stage(baseColorMap=G(D["base"]), opacityMap=G(D["opacity"]),
                                                    roughnessFactor=0.55, metallicFactor=0)],
                              translucent=True, translucentZWrite=True, translucentRecvShadows=True)
    with open(os.path.join(VDIR, "main.materials.json"), "w") as f:
        json.dump(mats, f, indent=1)
    return mats


def write_preview_json(T, D):
    prev = {}
    for atlas, mname in (("upper", "fd25_paint_upper"), ("arm", "fd25_paint_arm"), ("att", "fd25_attachment"),
                         ("uc", "fd25_undercarriage"), ("int", "fd25_interior")):
        prev[mname] = PREVIEW_MAT[atlas]
    for key, mname in (("steel", "fd25_steel"), ("steel_dark", "fd25_steel_dark"), ("track", "fd25_track"),
                       ("grating", "fd25_grating"), ("chrome", "fd25_chrome"), ("rubber", "fd25_rubber"),
                       ("black", "fd25_black")):
        t = T[key]
        prev[mname] = {k: v[1] for k, v in t.items() if k in ("base", "rough", "metal", "normal")}
    prev["fd25_decals"] = PREVIEW_MAT["decals"]
    with open(os.path.join(BUILD, "textures_preview.json"), "w") as f:
        json.dump(prev, f, indent=1)


def main():
    t0 = time.time()
    index = os.path.join(BUILD, "texture_index.json")
    if "materials" in sys.argv[1:]:
        # re-write main.materials.json / preview json from a previous full run
        with open(index) as f:
            idx = json.load(f)
        PREVIEW_MAT.update(idx["preview"])
        write_materials(idx["P"], idx["T"], idx["D"])
        write_preview_json(idx["T"], idx["D"])
        print("materials rewritten")
        return
    only = set(a for a in sys.argv[1:] if a in ("upper", "arm", "att", "uc", "int", "tiling", "decals"))
    prev = {}
    if only and os.path.exists(index):
        with open(index) as f:
            prev = json.load(f)
        PREVIEW_MAT.update(prev["preview"])
    want = lambda k: not only or k in only
    P = dict(prev.get("P", {}))
    if want("upper"):
        P["upper"] = paint_atlas("upper", 101, mud_height=1.25, wear_strength=1.0, dust=1.1)
    if want("arm"):
        P["arm"] = paint_atlas("arm", 202, mud_height=1.6, wear_strength=1.2, dust=1.0)
    if want("att"):
        P["att"] = solid_atlas("att", (0.06, 0.06, 0.065), 303, HI, wear_strength=1.5, mud_height=1.0, dust=0.55, extra="ring")
    if want("uc"):
        P["uc"] = solid_atlas("uc", (0.045, 0.045, 0.047), 404, LO, wear_strength=1.1, mud_height=0.75, dust=0.75)
    if want("int"):
        P["int"] = interior_atlas()
    T = tiling_textures() if want("tiling") else prev["T"]
    D = decal_atlas() if want("decals") else prev["D"]
    with open(index, "w") as f:
        json.dump(dict(P=P, T=T, D=D, preview=PREVIEW_MAT), f, indent=1)
    write_materials(P, T, D)
    write_preview_json(T, D)
    sizes = sum(os.path.getsize(os.path.join(TEX, f)) for f in os.listdir(TEX))
    print(f"textures written: {len(os.listdir(TEX))} files, {sizes / 1e6:.1f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
