"""
Qt Quick 3D geometry for the VR → flat un-warp.

`UnwarpGeometry` is a custom QQuick3DGeometry whose vertex UVs bake the
crop + un-warp (computed by vr_unwarp.build_unwarp_mesh). A Model using this
geometry, textured with the live video, renders the flat forward view as a
plain textured mesh — no fragment shader, so no `qsb` baking is needed and the
8K frame stays on the GPU.

Call rebuild() whenever the format/eye/FOV or the viewport aspect changes
(cheap — a few hundred vertices).
"""
import struct

from PyQt6.QtQuick3D import QQuick3DGeometry
from PyQt6.QtGui import QVector3D
from PyQt6.QtCore import QByteArray

import vr_unwarp as vu

_F32 = QQuick3DGeometry.Attribute.ComponentType.F32Type
_U32 = QQuick3DGeometry.Attribute.ComponentType.U32Type
_POS = QQuick3DGeometry.Attribute.Semantic.PositionSemantic
_UV  = QQuick3DGeometry.Attribute.Semantic.TexCoordSemantic
_IDX = QQuick3DGeometry.Attribute.Semantic.IndexSemantic
_TRIANGLES = QQuick3DGeometry.PrimitiveType.Triangles


class UnwarpGeometry(QQuick3DGeometry):
    """Grid mesh on the [-1,1]² plane whose UVs sample the un-warped source.
    Interleaved vertex layout: [x, y, z, u, v] float32 (stride 20 bytes)."""

    STRIDE = 5 * 4   # 3 position + 2 texcoord floats

    def __init__(self, parent=None):
        super().__init__(parent)
        self._desc = ""
        # Sensible default until the player calls rebuild() with the real format.
        self.rebuild(projection=vu.PROJ_EQUIRECT_180, hfov_deg=90.0,
                     out_aspect=16.0 / 9.0, eye='mono')

    def describe(self) -> str:
        return self._desc

    def rebuild(self, *, projection, hfov_deg, out_aspect, lens_fov_deg=200.0,
                eye='mono', cols=160, rows=160, flip_v=False,
                out_proj=vu.OUT_RECTILINEAR):
        """Recompute the mesh for the given format/eye/FOV/aspect/output-proj."""
        positions, uvs, indices = vu.build_unwarp_mesh(
            cols, rows, projection=projection, hfov_deg=hfov_deg,
            out_aspect=out_aspect, lens_fov_deg=lens_fov_deg, eye=eye,
            flip_v=flip_v, out_proj=out_proj)
        nverts = len(positions) // 3

        vbuf = bytearray(nverts * self.STRIDE)
        for k in range(nverts):
            struct.pack_into('<5f', vbuf, k * self.STRIDE,
                             positions[3 * k], positions[3 * k + 1],
                             positions[3 * k + 2], uvs[2 * k], uvs[2 * k + 1])
        ibuf = struct.pack('<%dI' % len(indices), *indices)

        self.clear()
        self.setStride(self.STRIDE)
        self.setVertexData(QByteArray(bytes(vbuf)))
        self.setIndexData(QByteArray(ibuf))
        self.setPrimitiveType(_TRIANGLES)
        self.addAttribute(_POS, 0, _F32)
        self.addAttribute(_UV, 3 * 4, _F32)
        self.addAttribute(_IDX, 0, _U32)
        self.setBounds(QVector3D(-1.0, -1.0, 0.0), QVector3D(1.0, 1.0, 0.0))
        self._index_count = len(indices)
        self._vertex_count = nverts
        self._desc = (f"{projection} eye={eye} hfov={hfov_deg:.0f} "
                      f"lens={lens_fov_deg:.0f} {cols}x{rows} "
                      f"({nverts} verts)")
        self.update()

    # exposed for tests
    def vertex_count(self) -> int:
        return getattr(self, '_vertex_count', 0)

    def index_count(self) -> int:
        return getattr(self, '_index_count', 0)
