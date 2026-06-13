"""
Unit tests for the VR → flat un-warp geometry (vr_unwarp.py).

These pin the mapping that turns a curved VR projection into a flat forward
view: the straight-ahead ray hits the centre of the source, rays aim the right
way (right→right, up→up), out-of-FOV rays are rejected, and the per-eye fold
lands in the correct half. Pure math — no Qt.
"""
import math
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import vr_unwarp as vu


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ── ray direction ─────────────────────────────────────────────────────────────
def test_forward_ray_is_centre():
    dx, dy, dz = vu.ray_direction(0.0, 0.0, 90.0, 16 / 9)
    assert _approx(dx, 0) and _approx(dy, 0) and _approx(dz, 1.0)


def test_ray_points_right_and_up():
    dx, dy, dz = vu.ray_direction(1.0, 1.0, 90.0, 1.0)
    assert dx > 0 and dy > 0 and dz > 0           # +x right, +y up, forward


def test_rectilinear_matches_old_pinhole_formula():
    # The refactored radial rectilinear branch must be identical to the old
    # normalize(nx*tan, ny*tan/asp, 1) pinhole formula (so nothing shifted).
    nx, ny, hfov, asp = 0.4, -0.7, 95.0, 16 / 9
    T = math.tan(math.radians(hfov) / 2)
    dx, dy, dz = nx * T, ny * T / asp, 1.0
    m = math.sqrt(dx * dx + dy * dy + dz * dz)
    got = vu.ray_direction(nx, ny, hfov, asp, vu.OUT_RECTILINEAR)
    for a, b in zip(got, (dx / m, dy / m, dz / m)):
        assert _approx(a, b, 1e-9)


def test_output_projections_agree_at_edge_differ_inside():
    hfov, asp = 120.0, 1.0
    # both output projections place the horizontal edge at exactly hfov/2 off-axis
    for op in (vu.OUT_RECTILINEAR, vu.OUT_STEREOGRAPHIC):
        _, _, dz = vu.ray_direction(1.0, 0.0, hfov, asp, op)
        assert _approx(math.degrees(math.acos(dz)), hfov / 2, 1e-6), op
    # …but distribute the interior differently (that's the anti-stretch effect)
    ar = math.degrees(math.acos(vu.ray_direction(0.5, 0.0, hfov, asp, vu.OUT_RECTILINEAR)[2]))
    as_ = math.degrees(math.acos(vu.ray_direction(0.5, 0.0, hfov, asp, vu.OUT_STEREOGRAPHIC)[2]))
    assert abs(ar - as_) > 1.0
    # both keep the forward ray dead-centre
    assert vu.ray_direction(0.0, 0.0, hfov, asp, vu.OUT_STEREOGRAPHIC) == (0.0, 0.0, 1.0)


def test_mesh_output_projection_changes_uvs():
    common = dict(projection=vu.PROJ_EQUIRECT_180, hfov_deg=120, out_aspect=1.0, eye='mono')
    _, uv_rect, _ = vu.build_unwarp_mesh(8, 8, out_proj=vu.OUT_RECTILINEAR, **common)
    _, uv_ster, _ = vu.build_unwarp_mesh(8, 8, out_proj=vu.OUT_STEREOGRAPHIC, **common)
    # centre vertex identical, but off-centre UVs differ between the two views
    assert uv_rect != uv_ster


# ── equirectangular ───────────────────────────────────────────────────────────
def test_equirect_forward_hits_centre():
    for proj in (vu.PROJ_EQUIRECT_180, vu.PROJ_EQUIRECT_360):
        u, v = vu.output_to_source_uv(0.0, 0.0, projection=proj, hfov_deg=90,
                                      out_aspect=1.0, eye='mono')
        assert _approx(u, 0.5) and _approx(v, 0.5), proj


def test_equirect_right_ray_moves_right_in_source():
    u, _ = vu.output_to_source_uv(0.6, 0.0, projection=vu.PROJ_EQUIRECT_180,
                                  hfov_deg=90, out_aspect=1.0, eye='mono')
    assert u > 0.5, "a ray aimed right should sample right-of-centre"


def test_equirect_up_ray_moves_up_in_source():
    # +y (up) ray → smaller v (top of image)
    _, v = vu.output_to_source_uv(0.0, 0.6, projection=vu.PROJ_EQUIRECT_180,
                                  hfov_deg=90, out_aspect=1.0, eye='mono')
    assert v < 0.5, "a ray aimed up should sample above-centre (smaller v)"


def test_equirect180_rejects_directions_outside_the_hemisphere():
    # A forward pinhole camera can only ever see the forward hemisphere, so it
    # always maps IN range. But a direction past 90° off-axis (only reachable by
    # looking around, not from the fixed forward camera) IS outside the 180
    # capture and must be rejected.
    a = math.radians(100)                          # 100° off the forward axis
    dx, dy, dz = math.sin(a), 0.0, math.cos(a)     # dz < 0
    assert vu.direction_to_source_uv(dx, dy, dz, vu.PROJ_EQUIRECT_180) is None
    # …and a forward ray is always fine, even at a very wide output FOV.
    assert vu.output_to_source_uv(0.9, 0.0, projection=vu.PROJ_EQUIRECT_180,
                                  hfov_deg=160, out_aspect=1.0, eye='mono',
                                  clamp=False) is not None


def test_equirect360_wraps_full_circle():
    # 360 never rejects: even a hard-right ray maps inside [0,1]
    dx, dy, dz = vu.ray_direction(1.0, 0.0, 179.0, 1.0)
    uv = vu.direction_to_source_uv(dx, dy, dz, vu.PROJ_EQUIRECT_360)
    assert uv is not None and 0.0 <= uv[0] <= 1.0


# ── fisheye ───────────────────────────────────────────────────────────────────
def test_fisheye_forward_hits_centre():
    u, v = vu.output_to_source_uv(0.0, 0.0, projection=vu.PROJ_FISHEYE,
                                  hfov_deg=90, out_aspect=1.0,
                                  lens_fov_deg=200, eye='mono')
    assert _approx(u, 0.5) and _approx(v, 0.5)


def test_fisheye_radius_grows_with_angle():
    # a ray 20° off-axis should land closer to centre than one 40° off-axis
    def radius(angle_deg):
        a = math.radians(angle_deg)
        dx, dy, dz = math.sin(a), 0.0, math.cos(a)
        u, v = vu.direction_to_source_uv(dx, dy, dz, vu.PROJ_FISHEYE, 200.0)
        return math.hypot(u - 0.5, v - 0.5)
    assert radius(20) < radius(40) < radius(80)


def test_fisheye_rejects_beyond_lens_fov():
    a = math.radians(110)                       # 110° off-axis, lens is 200° (±100°)
    dx, dy, dz = math.sin(a), 0.0, math.cos(a)
    assert vu.direction_to_source_uv(dx, dy, dz, vu.PROJ_FISHEYE, 200.0) is None


# ── eye fold ──────────────────────────────────────────────────────────────────
def test_apply_eye_halves():
    assert vu.apply_eye(0.5, 0.5, 'left')   == (0.25, 0.5)
    assert vu.apply_eye(0.5, 0.5, 'right')  == (0.75, 0.5)
    assert vu.apply_eye(0.5, 0.5, 'top')    == (0.5, 0.25)
    assert vu.apply_eye(0.5, 0.5, 'bottom') == (0.5, 0.75)
    assert vu.apply_eye(0.5, 0.5, 'mono')   == (0.5, 0.5)


def test_full_pipeline_left_eye_forward():
    # forward ray, left eye → centre of the LEFT half
    u, v = vu.output_to_source_uv(0.0, 0.0, projection=vu.PROJ_FISHEYE,
                                  hfov_deg=90, out_aspect=1.0,
                                  lens_fov_deg=200, eye='left')
    assert _approx(u, 0.25) and _approx(v, 0.5)


# ── clamp behaviour ───────────────────────────────────────────────────────────
def test_clamp_keeps_corner_rays_in_range():
    # A wide output FOV pushes the corners past a NARROW fisheye lens; with clamp
    # the UV stays valid (no None) so every mesh vertex has a usable coordinate,
    # and without clamp the same corner ray is rejected.
    kw = dict(projection=vu.PROJ_FISHEYE, hfov_deg=160, out_aspect=1.0,
              lens_fov_deg=120, eye='mono')
    u, v = vu.output_to_source_uv(1.0, 1.0, clamp=True, **kw)
    assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
    assert vu.output_to_source_uv(1.0, 1.0, clamp=False, **kw) is None


# ── mesh builder ──────────────────────────────────────────────────────────────
def test_build_unwarp_mesh_shapes():
    cols, rows = 8, 6
    pos, uv, idx = vu.build_unwarp_mesh(cols, rows, projection=vu.PROJ_EQUIRECT_180,
                                        hfov_deg=90, out_aspect=16 / 9, eye='left')
    nverts = (cols + 1) * (rows + 1)
    assert len(pos) == nverts * 3
    assert len(uv) == nverts * 2
    assert len(idx) == cols * rows * 6
    # every index is in range, every uv in [0,1]
    assert max(idx) < nverts and min(idx) >= 0
    assert all(0.0 <= c <= 1.0 for c in uv)
    # corners of the plane are at ±1
    assert pos[0:3] == [-1.0, -1.0, 0.0]            # first vertex bottom-left


def test_build_unwarp_mesh_centre_vertex_is_eye_centre():
    # even cols/rows → there is a vertex exactly at (0,0); its UV is the eye centre
    cols = rows = 8
    pos, uv, _ = vu.build_unwarp_mesh(cols, rows, projection=vu.PROJ_FISHEYE,
                                      hfov_deg=90, out_aspect=1.0,
                                      lens_fov_deg=200, eye='left')
    nverts_x = cols + 1
    centre = (rows // 2) * nverts_x + (cols // 2)
    cu, cv = uv[centre * 2], uv[centre * 2 + 1]
    assert _approx(cu, 0.25, 1e-9) and _approx(cv, 0.5, 1e-9)


def test_build_unwarp_mesh_flip_v():
    pos, uv, _ = vu.build_unwarp_mesh(4, 4, projection=vu.PROJ_EQUIRECT_360,
                                      hfov_deg=90, out_aspect=1.0, eye='mono',
                                      flip_v=True)
    pos2, uv2, _ = vu.build_unwarp_mesh(4, 4, projection=vu.PROJ_EQUIRECT_360,
                                        hfov_deg=90, out_aspect=1.0, eye='mono',
                                        flip_v=False)
    # v is mirrored about 0.5 between the two
    for k in range(0, len(uv), 2):
        assert _approx(uv[k], uv2[k])               # u unchanged
        assert _approx(uv[k + 1], 1.0 - uv2[k + 1])  # v flipped


def test_build_unwarp_mesh_rejects_degenerate():
    import pytest
    with pytest.raises(ValueError):
        vu.build_unwarp_mesh(0, 4, projection=vu.PROJ_EQUIRECT_180,
                             hfov_deg=90, out_aspect=1.0)


# ── CPU remap (thumbnails + hover preview) ────────────────────────────────────
def test_detect_stereo_eye():
    assert vu.detect_stereo_eye(7680, 3840) == 'left'    # 2:1 8K SBS
    assert vu.detect_stereo_eye(3840, 1920) == 'left'
    assert vu.detect_stereo_eye(1920, 1080) is None      # normal 16:9
    assert vu.detect_stereo_eye(4096, 1716) is None      # 2.39 cinema
    assert vu.detect_stereo_eye(0, 0) is None


def test_build_remap_shapes_bounds_and_centre():
    import numpy as np
    src_w, src_h, ow, oh = 8000, 4000, 65, 37          # odd → exact centre pixel
    mx, my = vu.build_remap(src_w, src_h, ow, oh, projection=vu.PROJ_EQUIRECT_180,
                            eye='left', hfov_deg=220, out_proj=vu.OUT_STEREOGRAPHIC)
    assert mx.shape == (oh, ow) and my.shape == (oh, ow)
    assert mx.dtype == np.float32 and my.dtype == np.float32
    assert mx.min() >= 0.0 and mx.max() <= src_w - 1
    assert my.min() >= 0.0 and my.max() <= src_h - 1
    # centre output pixel → centre of the LEFT eye (u≈0.25, v≈0.5)
    cy, cx = oh // 2, ow // 2
    assert _approx(mx[cy, cx], 0.25 * (src_w - 1), 1.0)
    assert _approx(my[cy, cx], 0.50 * (src_h - 1), 1.0)


def test_build_remap_left_eye_samples_left_half_only():
    src_w, src_h = 8000, 4000
    mx, _ = vu.build_remap(src_w, src_h, 80, 45, projection=vu.PROJ_EQUIRECT_180,
                           eye='left', hfov_deg=180, out_proj=vu.OUT_STEREOGRAPHIC)
    assert mx.max() <= 0.5 * (src_w - 1) + 1.0           # never crosses into the right eye


def test_frame_unwarper_outputs_flat_16_9_left_eye():
    import numpy as np
    import vr_frame
    src = np.zeros((400, 800, 3), np.uint8)              # 2:1 side-by-side source
    src[:, :400] = (10, 20, 30)                          # LEFT eye colour
    src[:, 400:] = (200, 100, 50)                        # RIGHT eye colour
    fu = vr_frame.FrameUnwarper(eye='left')
    out = fu.apply(src)
    assert out.shape[0] == 400
    assert abs(out.shape[1] / out.shape[0] - 16 / 9) < 0.02
    # centre of the flat view comes from the LEFT eye (not the right)
    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    assert tuple(int(c) for c in out[cy, cx]) == (10, 20, 30)
    # the remap is cached for a repeated same-size frame
    k1 = fu._key
    fu.apply(src)
    assert fu._key == k1


def test_frame_unwarper_for_path_detects_projection():
    import vr_frame
    assert vr_frame.for_path("clip_MKX200_LR.mp4")._params[0] == vu.PROJ_FISHEYE
    assert vr_frame.for_path("scene_180.mp4")._params[0] == vu.PROJ_EQUIRECT_180


# ── filename projection / lens detection ──────────────────────────────────────
def test_detect_projection_prioritises_180_and_tokens():
    # the user's examples
    assert vu.detect_projection("Studio_LR_180.mp4")     == vu.PROJ_EQUIRECT_180
    assert vu.detect_projection("clip_180x180_3dh.mp4")  == vu.PROJ_EQUIRECT_180
    assert vu.detect_projection("scene_fisheye190.mp4")  == vu.PROJ_FISHEYE
    # 180 wins over a stray '360' elsewhere in the name (the reported bug)
    assert vu.detect_projection("vid_180_lr_at_360fps.mp4") == vu.PROJ_EQUIRECT_180
    # a genuine 360 (no 180 token)
    assert vu.detect_projection("Scene_360_TB_8K.mp4")   == vu.PROJ_EQUIRECT_360
    assert vu.detect_projection("vr360_mono.mp4")        == vu.PROJ_EQUIRECT_360
    # digits inside a resolution / bitrate must NOT be read as a projection token
    assert vu.detect_projection("movie_1800kbps_lr.mp4") == vu.PROJ_EQUIRECT_180   # default
    assert vu.detect_projection("clip_3600kbps_360.mp4") == vu.PROJ_EQUIRECT_360   # 360 token, 3600 ignored
    assert vu.detect_projection("plain_movie.mp4")       == vu.PROJ_EQUIRECT_180   # default


def test_detect_projection_ignores_the_word_fishing():
    # 'fishing' / 'selfish' contain 'fish' but are NOT fisheye
    assert vu.detect_projection("Gone_Fishing_8K.mp4") == vu.PROJ_EQUIRECT_180
    assert vu.detect_projection("selfish_clip.mp4")    == vu.PROJ_EQUIRECT_180
    # real fisheye tags (fish + a FOV number, or 'fisheye') still match
    assert vu.detect_projection("clip_fisheye190.mp4")   == vu.PROJ_FISHEYE
    assert vu.detect_projection("clip_fish190.mp4")      == vu.PROJ_FISHEYE
    assert vu.detect_projection("scene_fish_200_lr.mp4") == vu.PROJ_FISHEYE


def test_halves_look_stereo_distinguishes_sbs_from_2d():
    import numpy as np
    yy, xx = np.mgrid[0:120, 0:100].astype(np.float32)
    eye = np.stack([(xx * 2) % 256, (yy * 2) % 256, (xx + yy) % 256],
                   axis=2).astype(np.uint8)               # structured "eye"
    # SBS pair: identical halves, and a small-parallax (4px) variant → stereo
    assert vu.halves_look_stereo(np.concatenate([eye, eye], axis=1)) is True
    assert vu.halves_look_stereo(
        np.concatenate([eye, np.roll(eye, 4, axis=1)], axis=1)) is True
    # 2D 2:1: left is a L→R gradient, right its mirror (anti-correlated) → NOT stereo
    g = np.tile(np.linspace(0, 255, 100, dtype=np.float32), (120, 1))
    left = np.stack([g, g, g], axis=2).astype(np.uint8)
    right = left[:, ::-1].copy()
    assert vu.halves_look_stereo(np.concatenate([left, right], axis=1)) is False
    # a flat/blank frame can't be judged → assume VR (never breaks detection)
    assert vu.halves_look_stereo(np.zeros((100, 200, 3), np.uint8)) is True


def test_detect_lens_fov_from_tags():
    assert vu.detect_lens_fov("scene_fisheye190.mp4") == 190.0
    assert vu.detect_lens_fov("clip_MKX200.mp4")      == 200.0
    assert vu.detect_lens_fov("x_VRCA220_8k.mp4")     == 220.0
    assert vu.detect_lens_fov("y_F180_fish.mp4")      == 180.0
    assert vu.detect_lens_fov("untagged.mp4")         == 200.0
