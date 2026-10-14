import ImageIO
import Vision

/// 基于 Vision `VNDetectHumanBodyPoseRequest` 的人体姿态检测器。
///
/// 输出 19 个关键点（归一化坐标，原点左下），用于画骨架；
/// 同时由关键点外接框推导出人物方框。
final class PoseDetector {

    private let request = VNDetectHumanBodyPoseRequest()

    /// 单个关键点的最低置信度，低于此值不画。
    private let minJointConfidence: Float = 0.25
    /// 至少要凑够多少个关键点才算一个人，避免误报。
    private let minJointCount = 4

    private static let jointMap: [VNHumanBodyPoseObservation.JointName: Joint] = [
        .nose: .nose,
        .neck: .neck,
        .root: .root,
        .leftShoulder: .leftShoulder,
        .rightShoulder: .rightShoulder,
        .leftElbow: .leftElbow,
        .rightElbow: .rightElbow,
        .leftWrist: .leftWrist,
        .rightWrist: .rightWrist,
        .leftHip: .leftHip,
        .rightHip: .rightHip,
        .leftKnee: .leftKnee,
        .rightKnee: .rightKnee,
        .leftAnkle: .leftAnkle,
        .rightAnkle: .rightAnkle,
        .leftEye: .leftEye,
        .rightEye: .rightEye,
        .leftEar: .leftEar,
        .rightEar: .rightEar,
    ]

    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation) -> [DetectedItem] {

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }

        guard let observations = request.results else { return [] }
        return observations.compactMap { makeItem(from: $0) }
    }

    private func makeItem(from observation: VNHumanBodyPoseObservation) -> DetectedItem? {
        guard let recognized = try? observation.recognizedPoints(.all) else { return nil }

        var joints: [Joint: CGPoint] = [:]
        var confidences: [Joint: Float] = [:]

        var minX: CGFloat = 1
        var minY: CGFloat = 1
        var maxX: CGFloat = 0
        var maxY: CGFloat = 0
        var confidenceSum: Float = 0

        for (visionName, point) in recognized {
            guard point.confidence >= minJointConfidence,
                  let joint = PoseDetector.jointMap[visionName] else { continue }

            joints[joint] = point.location
            confidences[joint] = point.confidence
            confidenceSum += point.confidence

            minX = min(minX, point.location.x)
            maxX = max(maxX, point.location.x)
            minY = min(minY, point.location.y)
            maxY = max(maxY, point.location.y)
        }

        guard joints.count >= minJointCount else { return nil }

        // 关键点外接框略微外扩，避免框贴得太紧
        let padX = (maxX - minX) * 0.14
        let padY = (maxY - minY) * 0.08
        let left = max(0, minX - padX)
        let bottom = max(0, minY - padY)
        let right = min(1, maxX + padX)
        let top = min(1, maxY + padY)

        guard right > left, top > bottom else { return nil }

        let box = CGRect(x: left, y: bottom, width: right - left, height: top - bottom)
        let average = confidenceSum / Float(joints.count)

        return DetectedItem(label: "人物",
                            confidence: average,
                            box: box,
                            joints: joints,
                            jointConfidences: confidences)
    }
}
