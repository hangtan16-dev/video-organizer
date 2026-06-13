// VR → flat un-warp view (Qt Quick 3D).
//
// The live video renders into a (covered) VideoOutput, which feeds a Texture
// via sourceItem; that texture is sampled by the un-warp Model whose geometry
// (set from Python) bakes the crop + reprojection into its UVs. No fragment
// shader → no qsb. The frame stays on the GPU the whole way.
import QtQuick
import QtQuick3D
import QtMultimedia

Item {
    id: root
    property bool flipV: false                  // toggle if the image is upside-down

    // Rendered (so it can be a texture source) but fully covered by the View3D.
    // Python connects the QMediaPlayer to this item via setVideoOutput(videoOut)
    // — passing the ITEM (not its videoSink, which marshals to a raw voidptr).
    VideoOutput {
        id: vout
        objectName: "videoOut"
        anchors.fill: parent
        fillMode: VideoOutput.Stretch           // video fills [0,1]×[0,1] of the texture
    }

    View3D {
        id: view
        anchors.fill: parent
        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "black"
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
        }

        // Orthographic so the flat [-1,1]² plane maps straight to the viewport.
        OrthographicCamera {
            id: cam
            z: 600
        }

        Model {
            id: unwarpModel
            objectName: "unwarpModel"
            // Scale the [-1,1]² mesh to fill the viewport (1 ortho unit ≈ 1 px).
            scale: Qt.vector3d(view.width / 2.0, view.height / 2.0, 1.0)
            materials: PrincipledMaterial {
                lighting: PrincipledMaterial.NoLighting
                cullMode: Material.NoCulling
                baseColorMap: Texture {
                    sourceItem: vout
                    flipV: root.flipV
                    minFilter: Texture.Linear
                    magFilter: Texture.Linear
                }
            }
            // `geometry` is assigned from Python (UnwarpGeometry).
        }
    }
}
