import SwiftUI

/// 底部控制面板：开关检测项、调置信度、切换摄像头。
struct ControlPanel: View {

    @ObservedObject var engine: DetectionEngine
    let usingFrontCamera: Bool
    let onSwitchCamera: () -> Void

    private let accent = Color(red: 0.15, green: 0.90, blue: 0.95)

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 8) {
                ToggleChip(title: "物体识别",
                           systemImage: "cube.transparent",
                           isOn: $engine.detectObjects,
                           tint: Color(red: 1.00, green: 0.78, blue: 0.16))

                ToggleChip(title: "姿态骨架",
                           systemImage: "figure.walk",
                           isOn: $engine.detectPose,
                           tint: accent)

                Spacer(minLength: 0)

                Button(action: onSwitchCamera) {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(width: 40, height: 34)
                        .background(Capsule().fill(Color.white.opacity(0.14)))
                }
                .buttonStyle(.plain)
            }

            HStack(spacing: 10) {
                Text("置信度")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.white.opacity(0.75))

                Slider(value: $engine.minConfidence, in: 0.10...0.90)
                    .tint(accent)

                Text("\(Int((engine.minConfidence * 100).rounded()))%")
                    .font(.system(size: 11, weight: .semibold))
                    .monospacedDigit()
                    .foregroundColor(.white)
                    .frame(width: 36, alignment: .trailing)
            }

            Text(hintText)
                .font(.system(size: 10))
                .foregroundColor(.white.opacity(0.55))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.black.opacity(0.55))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.12), lineWidth: 0.5)
        )
    }

    private var hintText: String {
        if !engine.isModelReady {
            return "模型未加载，物体识别不可用；姿态骨架仍可正常工作。"
        }
        switch (engine.detectObjects, engine.detectPose) {
        case (true, true):
            return "青色框 = 人物（带骨架）· 黄色框 = 其他物体\n全画面分块扫描，画面各处都能识别；小物体凑近一点更稳"
        case (true, false):
            return "黄色框 = 识别到的物体\n全画面分块扫描，画面各处都能识别；识别不到就把置信度调低试试"
        case (false, true):
            return "青色框 = 人物，橙线 = 骨骼连接"
        case (false, false):
            return "两个开关都关掉了，打开任意一个开始识别"
        }
    }
}

/// 胶囊形状的开关按钮。
struct ToggleChip: View {

    let title: String
    let systemImage: String
    @Binding var isOn: Bool
    let tint: Color

    var body: some View {
        Button {
            withAnimation(.easeOut(duration: 0.15)) {
                isOn.toggle()
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .bold))
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundColor(isOn ? .black : .white.opacity(0.85))
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(
                Capsule().fill(isOn ? tint : Color.white.opacity(0.14))
            )
        }
        .buttonStyle(.plain)
    }
}
