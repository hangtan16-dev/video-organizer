"""
VR → flat (rectilinear) un-warp math.

A VR video is a CURVED projection of the world:
  • equirectangular — longitude→x, latitude→y (180° half-sphere or full 360°), or
  • fisheye         — angle-from-axis→radius (a 180–220° lens, e.g. MKX200).
Playing it "flat" means simulating a normal pinhole camera looking straight
ahead: for every output pixel we shoot a ray, find where that ray hits the
source projection, and sample there. (Fixed forward view — no look-around — so
this mapping is STATIC and can be baked into a mesh's UVs once per video.)

This module is the pure geometry. It has NO Qt dependency so it can be unit
tested directly, and the same functions compute the per-vertex UVs of the
Qt Quick 3D un-warp mesh (see vr_unwarp_mesh.py).

Conventions:
  • Output coord (nx, ny) ∈ [-1, 1], centre = straight ahead, +x right, +y up.
  • Camera looks down +Z; +Y is up; +X is right (right-handed).
  • Source UV ∈ [0, 1], origin top-left (v grows downward, image convention).
  • Returned UV is for ONE eye's FULL frame; fold into an SBS/TB half with
    apply_eye().
"""
import math
import os
import re

# Projection kinds
PROJ_EQUIRECT_180 = 'equirect180'   # 180°H × 180°V half-equirectangular
PROJ_EQUIRECT_360 = 'equirect360'   # full 360°×180° equirectangular
PROJ_FISHEYE      = 'fisheye'       # circular fisheye, lens_fov_deg total FOV

_PROJECTIONS = (PROJ_EQUIRECT_180, PROJ_EQUIRECT_360, PROJ_FISHEYE)

# OUTPUT projection — how the flat view itself maps angles to the screen. This
# is the trade-off the viewer picks:
#   • rectilinear  — straight lines stay straight, but the edges STRETCH more
#     and more the wider the FOV (a pinhole/standard-lens look).
#   • stereographic — conformal: shapes stay natural (no edge stretch) even at a
#     very wide FOV, at the cost of straight lines bowing slightly.
OUT_RECTILINEAR   = 'rectilinear'
OUT_STEREOGRAPHIC = 'stereographic'

_OUT_PROJECTIONS = (OUT_RECTILINEAR, OUT_STEREOGRAPHIC)


def detect_projection(path: str) -> str:
    """Guess the source projection from the filename (the tag convention VR
    players use). Priority:
      1. a fisheye-lens tag (fisheye / MKX / VRCA / RF52 / F1xx) → fisheye,
      2. a '180' token (LR_180, 180x180, vr180, …) → 180 half-equirect — checked
         BEFORE 360 so a stray '360' (a resolution, bitrate or count) can't
         override a real 180 tag (the bug the user hit),
      3. a '360' token → full 360 equirect,
      4. otherwise default to 180 (what untagged SBS files are).
    '180'/'360' are matched as standalone tokens (not flanked by other digits),
    so '3600', '1800', '1080' etc. don't count."""
    n = os.path.basename(path).lower()
    # 'fisheye' as a whole word, or 'fish' ONLY when followed by a FOV number
    # (fish190 / fish_200) — NOT inside ordinary words like "fishing".
    if re.search(r'fisheye|fish[ _\-]?\d|mkx|vrca|rf52|_f1[789]\d|_f2[0-2]\d', n):
        return PROJ_FISHEYE
    if re.search(r'(?<!\d)180(?!\d)', n):
        return PROJ_EQUIRECT_180
    if re.search(r'(?<!\d)360(?!\d)', n):
        return PROJ_EQUIRECT_360
    return PROJ_EQUIRECT_180


def detect_lens_fov(path: str) -> float:
    """Guess a fisheye lens's total FOV (degrees) from the filename, e.g.
    fisheye190 → 190, MKX200 → 200, VRCA220 → 220, F180 → 180. Prefers a number
    attached to a lens tag; falls back to a standalone 160–230 token; else 200°."""
    n = os.path.basename(path).lower()
    m = re.search(r'(?:fisheye|fish|mkx|vrca|rf52|_f)[\s_\-]?(\d{3})', n)
    if m and 150 <= int(m.group(1)) <= 240:
        return float(m.group(1))
    for fov in (220, 210, 200, 190, 180):
        if re.search(rf'(?<!\d){fov}(?!\d)', n):
            return float(fov)
    return 200.0


def ray_direction(nx, ny, hfov_deg, out_aspect, out_proj=OUT_RECTILINEAR):
    """Unit ray for output coord (nx, ny) ∈ [-1,1], looking forward with
    horizontal FOV `hfov_deg`. `out_aspect` = output width/height (so the
    vertical FOV follows the window shape). `out_proj` chooses how the screen
    radius maps to the view angle (rectilinear vs stereographic).

    Both map the horizontal edge (nx=±1) to exactly hfov/2, so the FOV slider
    means the same thing for either; they differ only in how the interior is
    distributed — rectilinear stretches the edges, stereographic doesn't.
    (The rectilinear branch is mathematically identical to the old
    tan-of-angle pinhole formula.)"""
    ax = nx
    ay = ny / out_aspect
    r = math.hypot(ax, ay)
    if r == 0.0:
        return 0.0, 0.0, 1.0
    half = math.radians(hfov_deg) * 0.5
    if out_proj == OUT_STEREOGRAPHIC:
        theta = 2.0 * math.atan(r * math.tan(half * 0.5))   # r=1 → θ=half
    else:
        theta = math.atan(r * math.tan(half))               # r=1 → θ=half
    st = math.sin(theta)
    return st * (ax / r), st * (ay / r), math.cos(theta)


def direction_to_source_uv(dx, dy, dz, projection, lens_fov_deg=200.0):
    """Map a (roughly forward) unit ray to a FULL-frame source UV in [0,1], or
    None if the ray falls outside the captured field of view."""
    if projection == PROJ_FISHEYE:
        # angle from the forward (+Z) optical axis
        theta = math.acos(max(-1.0, min(1.0, dz)))
        max_theta = math.radians(lens_fov_deg) * 0.5
        if max_theta <= 0 or theta > max_theta:
            return None
        r = theta / max_theta                     # 0 at centre, 1 at lens edge
        phi = math.atan2(dy, dx)                   # azimuth around the axis
        u = 0.5 + 0.5 * r * math.cos(phi)
        v = 0.5 - 0.5 * r * math.sin(phi)          # flip: image v grows downward
        return u, v

    # equirectangular: forward = centre of the image
    lon = math.atan2(dx, dz)                       # azimuth (0 = forward)
    lat = math.asin(max(-1.0, min(1.0, dy)))       # elevation
    if projection == PROJ_EQUIRECT_360:
        u = 0.5 + lon / (2.0 * math.pi)
        v = 0.5 - lat / math.pi
    else:   # equirect180 — 180°H × 180°V, centred
        if abs(lon) > math.pi / 2 or abs(lat) > math.pi / 2:
            return None
        u = 0.5 + lon / math.pi
        v = 0.5 - lat / math.pi
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return None
    return u, v


def apply_eye(u, v, eye):
    """Fold a full-frame UV into the chosen eye's half of a packed stereo frame.
    eye: 'left'/'right' (side-by-side), 'top'/'bottom' (over-under), else mono."""
    if eye == 'left':   return u * 0.5, v
    if eye == 'right':  return 0.5 + u * 0.5, v
    if eye == 'top':    return u, v * 0.5
    if eye == 'bottom': return u, 0.5 + v * 0.5
    return u, v


def output_to_source_uv(nx, ny, *, projection, hfov_deg, out_aspect,
                        lens_fov_deg=200.0, eye='mono', clamp=True,
                        out_proj=OUT_RECTILINEAR):
    """Full pipeline: output coord (nx,ny ∈ [-1,1]) → source UV ∈ [0,1] for the
    chosen eye. Returns None if the ray is outside the captured FOV and
    clamp=False; if clamp=True, the UV is clamped to the valid edge instead
    (so a mesh vertex always has a usable, if stretched, coordinate)."""
    dx, dy, dz = ray_direction(nx, ny, hfov_deg, out_aspect, out_proj)
    uv = direction_to_source_uv(dx, dy, dz, projection, lens_fov_deg)
    if uv is None:
        if not clamp:
            return None
        # Re-derive an edge-clamped UV so corner rays beyond the FOV still map
        # to the nearest captured pixel rather than vanishing.
        uv = _clamped_uv(dx, dy, dz, projection, lens_fov_deg)
    u, v = uv
    return apply_eye(u, v, eye)


def build_unwarp_mesh(cols, rows, *, projection, hfov_deg, out_aspect,
                      lens_fov_deg=200.0, eye='mono', flip_v=False,
                      out_proj=OUT_RECTILINEAR):
    """Build a (cols+1)×(rows+1) grid that bakes the crop+un-warp into its UVs.

    Returns (positions, uvs, indices):
      • positions — flat [x,y,z, …] on the output plane, x,y ∈ [-1,1], z=0
        (an orthographic camera maps this square to the viewport),
      • uvs       — flat [u,v, …] source-texture coords (crop + un-warp baked in),
      • indices   — flat triangle indices (two per cell).
    The image distortion lives entirely in the UVs, so rendering is just a
    normal textured mesh — no shader, no qsb. Denser grid ⇒ smoother un-warp.
    `flip_v` flips the V coordinate for renderers whose texture origin is at the
    bottom (set by the player if the image comes out upside-down)."""
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be >= 1")
    positions = []
    uvs = []
    nverts_x = cols + 1
    for j in range(rows + 1):
        ny = -1.0 + 2.0 * j / rows
        for i in range(nverts_x):
            nx = -1.0 + 2.0 * i / cols
            positions.extend((nx, ny, 0.0))
            u, v = output_to_source_uv(
                nx, ny, projection=projection, hfov_deg=hfov_deg,
                out_aspect=out_aspect, lens_fov_deg=lens_fov_deg,
                eye=eye, clamp=True, out_proj=out_proj)
            uvs.extend((u, 1.0 - v if flip_v else v))
    indices = []
    for j in range(rows):
        for i in range(cols):
            v0 = j * nverts_x + i
            v1 = v0 + 1
            v2 = v0 + nverts_x
            v3 = v2 + 1
            indices.extend((v0, v2, v1, v1, v2, v3))
    return positions, uvs, indices


def _clamped_uv(dx, dy, dz, projection, lens_fov_deg):
    if projection == PROJ_FISHEYE:
        phi = math.atan2(dy, dx)
        u = 0.5 + 0.5 * math.cos(phi)
        v = 0.5 - 0.5 * math.sin(phi)
        return u, v
    lon = math.atan2(dx, dz)
    lat = math.asin(max(-1.0, min(1.0, dy)))
    if projection == PROJ_EQUIRECT_360:
        u = 0.5 + lon / (2.0 * math.pi)
        v = 0.5 - lat / math.pi
    else:
        u = 0.5 + max(-math.pi / 2, min(math.pi / 2, lon)) / math.pi
        v = 0.5 - max(-math.pi / 2, min(math.pi / 2, lat)) / math.pi
    return min(1.0, max(0.0, u)), min(1.0, max(0.0, v))


def halves_look_stereo(frame, min_corr=0.6):
    """For a ~2:1 frame, True if the LEFT and RIGHT halves look like a stereo
    PAIR (nearly the same image — side-by-side VR) rather than two different
    parts of one 2D scene (e.g. a 2:1 'fishing' clip). Cheap normalized
    correlation on downscaled, mean-removed halves.

    CONSERVATIVE by design: a real SBS pair's halves correlate ~0.85–0.98, so a
    0.6 threshold virtually never rejects genuine VR; it only filters out a 2:1
    *2D* video whose halves are clearly different. On any error / blank frame it
    returns True (assume VR) so detection is never broken by this guard."""
    try:
        import numpy as np
        import cv2
        h, w = frame.shape[:2]
        half = w // 2
        if h < 4 or half < 4:
            return True
        s = 48

        def prep(x):
            g = x.astype(np.float32)
            if g.ndim == 3:
                g = g.mean(axis=2)
            g = cv2.resize(g, (s, s))
            return g - g.mean()

        left = prep(frame[:, :half])
        right = prep(frame[:, half:half * 2])
        denom = float(np.sqrt((left * left).sum()) * np.sqrt((right * right).sum()))
        if denom < 1e-6:
            return True                      # flat/blank → can't tell → assume VR
        return float((left * right).sum() / denom) >= min_corr
    except Exception:
        return True


def is_sbs_aspect(w, h):
    """True if a frame's shape is the tight ~2:1 side-by-side band (excludes
    16:9, 21:9 and 2.39 cinema). Aspect only — used where the resolution isn't
    full (e.g. a downscaled cached thumbnail, which preserves the source
    aspect)."""
    return w > 0 and h > 0 and 1.85 <= (w / h) <= 2.15


def detect_stereo_eye(w, h):
    """Frame-shape stereo detection from a FULL-resolution frame (player + hover
    preview): a ~2:1, high-res frame is side-by-side VR → 'left'; else None, so
    normal videos — incl. 21:9 / 2.39 cinema — are never un-warped. The width
    floor avoids treating a low-res 2:1 clip as VR."""
    if is_sbs_aspect(w, h) and w >= 3840:
        return 'left'
    return None


def build_remap(src_w, src_h, out_w, out_h, *, projection=PROJ_EQUIRECT_180,
                eye='left', hfov_deg=220.0, lens_fov_deg=200.0,
                out_proj=OUT_STEREOGRAPHIC):
    """Build OpenCV remap tables (map_x, map_y as float32 arrays of shape
    (out_h, out_w)) that un-warp a `src_w`×`src_h` VR frame into an
    `out_w`×`out_h` FLAT view — the same crop + reprojection the fullscreen
    player does on the GPU, but for the CPU-decoded thumbnail / hover-preview
    frames. Build it ONCE per (sizes + params) and reuse it with
    `cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)` on every frame.

    No vertical flip is applied here: a PyAV/cv2 frame already has a top-left
    origin, which matches this module's v=0=top convention (the flip the
    Quick3D path needs is only for its bottom-left texture origin)."""
    import numpy as np

    xs = np.linspace(-1.0, 1.0, out_w, dtype=np.float64)
    ys = np.linspace(1.0, -1.0, out_h, dtype=np.float64)   # top row = +1 (up)
    nx, ny = np.meshgrid(xs, ys)
    out_aspect = out_w / float(out_h)

    ax = nx
    ay = ny / out_aspect
    r = np.hypot(ax, ay)
    half = math.radians(hfov_deg) * 0.5
    if out_proj == OUT_STEREOGRAPHIC:
        theta = 2.0 * np.arctan(r * math.tan(half * 0.5))
    else:
        theta = np.arctan(r * math.tan(half))
    with np.errstate(invalid='ignore', divide='ignore'):
        ux = np.where(r > 0, ax / r, 0.0)
        uy = np.where(r > 0, ay / r, 0.0)
    st = np.sin(theta)
    dx = st * ux
    dy = st * uy
    dz = np.cos(theta)

    if projection == PROJ_FISHEYE:
        max_theta = math.radians(lens_fov_deg) * 0.5
        th = np.arccos(np.clip(dz, -1.0, 1.0))
        rr = np.clip(th / max_theta, 0.0, 1.0)
        phi = np.arctan2(dy, dx)
        u = 0.5 + 0.5 * rr * np.cos(phi)
        v = 0.5 - 0.5 * rr * np.sin(phi)
    else:
        lon = np.arctan2(dx, dz)
        lat = np.arcsin(np.clip(dy, -1.0, 1.0))
        if projection == PROJ_EQUIRECT_360:
            u = 0.5 + lon / (2.0 * math.pi)
            v = 0.5 - lat / math.pi
        else:   # equirect180 — clamp beyond the captured hemisphere to its edge
            u = 0.5 + np.clip(lon, -math.pi / 2, math.pi / 2) / math.pi
            v = 0.5 - np.clip(lat, -math.pi / 2, math.pi / 2) / math.pi
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    if eye == 'left':
        u = u * 0.5
    elif eye == 'right':
        u = 0.5 + u * 0.5
    elif eye == 'top':
        v = v * 0.5
    elif eye == 'bottom':
        v = 0.5 + v * 0.5

    map_x = (u * (src_w - 1)).astype(np.float32)
    map_y = (v * (src_h - 1)).astype(np.float32)
    return map_x, map_y
