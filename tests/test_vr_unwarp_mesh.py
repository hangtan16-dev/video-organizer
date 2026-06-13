"""
Construction tests for the Qt Quick 3D un-warp geometry.

Verifies the QQuick3DGeometry packs vertex/index data without error and reports
the right counts. (Actual on-GPU rendering can't be checked headlessly — that's
validated by eye via vr_unwarp_preview.py.)
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_ORG', 'VOrgTest')
os.environ.setdefault('VIDEO_ORGANIZER_QSETTINGS_APP', 'VRUnwarpMeshTest')

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


def test_geometry_constructs_with_default(qapp):
    import vr_unwarp as vu
    from vr_unwarp_mesh import UnwarpGeometry
    g = UnwarpGeometry()                      # __init__ does a default rebuild
    assert g.vertex_count() == 161 * 161
    assert g.index_count() == 160 * 160 * 6
    assert vu.PROJ_EQUIRECT_180 in g.describe()


def test_geometry_rebuild_changes_grid(qapp):
    import vr_unwarp as vu
    from vr_unwarp_mesh import UnwarpGeometry
    g = UnwarpGeometry()
    g.rebuild(projection=vu.PROJ_FISHEYE, hfov_deg=100.0, out_aspect=1.0,
              lens_fov_deg=200.0, eye='left', cols=10, rows=8)
    assert g.vertex_count() == 11 * 9
    assert g.index_count() == 10 * 8 * 6
    d = g.describe()
    assert 'fisheye' in d and 'eye=left' in d


def test_detect_projection_and_lens():
    import vr_unwarp as vu
    assert vu.detect_projection("clip_MKX200.mp4") == vu.PROJ_FISHEYE
    assert vu.detect_projection("scene_360_TB.mp4") == vu.PROJ_EQUIRECT_360
    assert vu.detect_projection("movie_180_LR.mp4") == vu.PROJ_EQUIRECT_180
    assert vu.detect_lens_fov("clip_MKX220.mp4") == 220.0
    assert vu.detect_lens_fov("clip_fisheye190.mp4") == 190.0
    assert vu.detect_lens_fov("untagged.mp4") == 200.0
