"""
Apply the VR → flat un-warp to CPU-decoded frames (thumbnails + hover preview),
reusing the SAME math as the fullscreen player (vr_unwarp.build_remap) so all
three look identical.

`FrameUnwarper` caches an OpenCV remap for a fixed source size + params and
applies it to every same-sized frame (cv2.remap), rebuilding only when the
source size changes. Channel order (RGB vs BGR) is preserved — it's a pure
spatial remap.

Output is a flat 16:9 view (one eye, stereographic 220° by default — the look
the user settled on in the player).
"""
import vr_unwarp as vu

OUT_ASPECT = 16.0 / 9.0


class FrameUnwarper:
    def __init__(self, *, projection=vu.PROJ_EQUIRECT_180, eye='left',
                 hfov_deg=220.0, lens_fov_deg=200.0,
                 out_proj=vu.OUT_STEREOGRAPHIC, out_aspect=OUT_ASPECT):
        self._params = (projection, eye, hfov_deg, lens_fov_deg, out_proj)
        self._out_aspect = out_aspect
        self._key = None
        self._map = None

    def apply(self, frame):
        """frame: H×W×3 uint8 ndarray. Returns a flat (out_h≈H, 16:9) ndarray of
        the same channel order. The remap is cached and only rebuilt if the
        source size changes."""
        import cv2
        h, w = frame.shape[:2]
        out_h = max(1, h)
        out_w = max(1, int(round(out_h * self._out_aspect)))
        key = (w, h, out_w, out_h)
        if key != self._key:
            proj, eye, hfov, lens, oproj = self._params
            self._map = vu.build_remap(w, h, out_w, out_h, projection=proj,
                                       eye=eye, hfov_deg=hfov, lens_fov_deg=lens,
                                       out_proj=oproj)
            self._key = key
        mx, my = self._map
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)


def for_path(video_path, eye='left'):
    """Build a FrameUnwarper with projection/lens auto-detected from the
    filename (defaults to equirect180, which is what untagged SBS files are)."""
    return FrameUnwarper(projection=vu.detect_projection(video_path), eye=eye,
                         lens_fov_deg=vu.detect_lens_fov(video_path))
