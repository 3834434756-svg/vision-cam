import AVFoundation
import SwiftUI
import UIKit

/// 把 `AVCaptureVideoPreviewLayer` 包进 SwiftUI。
///
/// `videoGravity` 用 `.resizeAspectFill`，与 `CoordinateMapper` 的换算方式一致，
/// 这样叠加层画出来的框才会严丝合缝地贴住画面里的物体。
struct CameraPreview: UIViewRepresentable {

    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
    }

    final class PreviewView: UIView {

        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            // layerClass 已经保证类型正确
            guard let layer = layer as? AVCaptureVideoPreviewLayer else {
                fatalError("layerClass 必须是 AVCaptureVideoPreviewLayer")
            }
            return layer
        }
    }
}
