import SwiftUI

struct ContentView: View {

    @StateObject private var camera = CameraManager()
    @StateObject private var engine = DetectionEngine()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            OverlayView(items: engine.items, imageSize: engine.orientedImageSize)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                statusBar
                Spacer(minLength: 0)
                ControlPanel(engine: engine,
                             usingFrontCamera: camera.usingFrontCamera,
                             onSwitchCamera: { camera.switchCamera() })
            }
            .padding(.horizontal, 14)
            .padding(.top, 8)
            .padding(.bottom, 18)

            if camera.authorizationDenied {
                permissionNotice
            }
        }
        .statusBarHidden(true)
        .onAppear {
            camera.onFrame = { pixelBuffer, orientation in
                engine.process(pixelBuffer: pixelBuffer, orientation: orientation)
            }
            camera.start()
        }
        .onDisappear {
            camera.onFrame = nil
            camera.stop()
            engine.reset()
        }
    }

    // MARK: - 顶部状态条

    private var statusBar: some View {
        HStack(spacing: 10) {
            StatusPill(icon: "speedometer",
                       text: String(format: "%.1f FPS", engine.fps))
            StatusPill(icon: "timer",
                       text: String(format: "%.0f ms", engine.latencyMilliseconds))
            StatusPill(icon: "viewfinder",
                       text: "\(engine.items.count) 个目标")

            Spacer(minLength: 0)

            if engine.modelErrorMessage != nil {
                StatusPill(icon: "exclamationmark.triangle.fill",
                           text: "模型异常",
                           tint: .orange)
            }
        }
    }

    // MARK: - 权限提示

    private var permissionNotice: some View {
        VStack(spacing: 14) {
            Image(systemName: "camera.fill")
                .font(.system(size: 40))
                .foregroundColor(.white.opacity(0.85))
            Text("需要相机权限")
                .font(.headline)
                .foregroundColor(.white)
            Text("请到「设置 → 隐私与安全性 → 相机」里允许「视觉相机」使用相机。\n所有画面均在本机处理，不会上传。")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundColor(.white.opacity(0.7))
        }
        .padding(28)
        .background(RoundedRectangle(cornerRadius: 20).fill(Color.black.opacity(0.82)))
        .padding(40)
    }
}

// MARK: - 状态胶囊

private struct StatusPill: View {

    let icon: String
    let text: String
    var tint: Color = .white

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .bold))
            Text(text)
                .font(.system(size: 11, weight: .semibold))
                .monospacedDigit()
        }
        .foregroundColor(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(Color.black.opacity(0.45)))
        .overlay(Capsule().stroke(Color.white.opacity(0.12), lineWidth: 0.5))
    }
}

#Preview {
    ContentView()
}
