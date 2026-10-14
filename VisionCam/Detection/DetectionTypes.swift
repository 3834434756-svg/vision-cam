import CoreGraphics
import Foundation

// MARK: - 骨架关键点

/// 人体姿态的 19 个关键点，与 Vision 的 `VNHumanBodyPoseObservation.JointName` 一一对应。
enum Joint: String, CaseIterable, Hashable {
    case nose, neck, root
    case leftShoulder, leftElbow, leftWrist
    case rightShoulder, rightElbow, rightWrist
    case leftHip, leftKnee, leftAnkle
    case rightHip, rightKnee, rightAnkle
    case leftEye, rightEye, leftEar, rightEar
}

/// 骨骼连接关系（用于画骨架连线）。
let skeletonBones: [(Joint, Joint)] = [
    (.nose, .neck),
    (.nose, .leftEye), (.leftEye, .leftEar),
    (.nose, .rightEye), (.rightEye, .rightEar),
    (.neck, .leftShoulder), (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
    (.neck, .rightShoulder), (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
    (.neck, .root),
    (.root, .leftHip), (.leftHip, .leftKnee), (.leftKnee, .leftAnkle),
    (.root, .rightHip), (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
]

// MARK: - 检测结果

/// 一个被检测到的目标。
///
/// 所有坐标都是 **Vision 归一化坐标**：原点在左下角，x/y 取值 0~1，
/// 对应「已按设备方向摆正」的图像。
struct DetectedItem: Identifiable {
    let id = UUID()

    /// 显示名称（中文）。
    let label: String
    /// 置信度 0~1。
    let confidence: Float
    /// 外接矩形（Vision 归一化坐标）。
    let box: CGRect
    /// 骨架关键点，仅人物有值。
    let joints: [Joint: CGPoint]
    /// 各关键点置信度。
    let jointConfidences: [Joint: Float]

    /// 是否为人（带骨架信息）。
    var isPerson: Bool { !joints.isEmpty }

    static func object(label: String, confidence: Float, box: CGRect) -> DetectedItem {
        DetectedItem(label: label, confidence: confidence, box: box, joints: [:], jointConfidences: [:])
    }
}
