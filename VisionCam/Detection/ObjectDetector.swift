import CoreML
import ImageIO
import Vision

/// 基于 YOLOv8n（CoreML）的物体检测器。
///
/// 模型是 Ultralytics 导出的 **带内置 NMS 的 CoreML pipeline**：
///   * 输入：`image`（640×640，RGB）
///   * 输出：`confidence` 形状 [N, 80]，`coordinates` 形状 [N, 4]
///
/// `coordinates` 每行是 `[cx, cy, w, h]`，**已归一化到 0~1**（据 Apple CoreML
/// 官方 proto：x/y 是盒子中心点，w/h 是宽高）。所以直接乘图像尺寸即可还原。
///
/// 注意：原始模型还带 `iouThreshold` / `confidenceThreshold` 两个必填标量输入，
/// 会导致 Vision 无法调用。本工程内置的 `YOLOv8n.mlpackage` 已打补丁把这两个
/// 输入移除，让 NMS 阶段回退使用内置参数（0.45 / 0.25），从而只剩单个 image 输入。
final class ObjectDetector {

    private static let modelName = "YOLOv8n"

    private let vnModel: VNCoreMLModel

    init?() {
        guard let model = ObjectDetector.loadModel(named: ObjectDetector.modelName),
              let vnModel = try? VNCoreMLModel(for: model) else {
            return nil
        }
        self.vnModel = vnModel
    }

    /// 优先加载 Xcode 编译好的 `.mlmodelc`；万一被当成普通资源打进包，
    /// 就退回运行时编译 `.mlpackage`。
    private static func loadModel(named name: String) -> MLModel? {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
           let model = try? MLModel(contentsOf: url, configuration: configuration) {
            return model
        }

        if let url = Bundle.main.url(forResource: name, withExtension: "mlpackage"),
           let compiled = try? MLModel.compileModel(at: url),
           let model = try? MLModel(contentsOf: compiled, configuration: configuration) {
            return model
        }

        return nil
    }

    /// 在给定帧上跑一次检测。
    ///
    /// - Parameters:
    ///   - pixelBuffer: 相机输出帧。
    ///   - orientation: 把该帧摆正所需的 EXIF 方向。
    ///   - minConfidence: 前端置信度门槛（模型内部门槛是 0.25，这里只做二次过滤）。
    ///   - skipPersonClass: 是否跳过 `person` 类。姿态检测已单独处理人物时置 true，避免重复画框。
    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation,
                minConfidence: Float,
                skipPersonClass: Bool) -> [DetectedItem] {

        let request = VNCoreMLRequest(model: vnModel)
        // scaleFill：把整幅图拉伸到 640×640。归一化坐标因此可以 1:1 映射回原图。
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }

        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            return []
        }

        var confidenceArray: MLMultiArray?
        var coordinateArray: MLMultiArray?
        for observation in observations {
            switch observation.featureName {
            case "confidence":
                confidenceArray = observation.featureValue.multiArrayValue
            case "coordinates":
                coordinateArray = observation.featureValue.multiArrayValue
            default:
                break
            }
        }

        guard let confidence = confidenceArray, let coordinates = coordinateArray else {
            return []
        }

        return ObjectDetector.parse(confidence: confidence,
                                    coordinates: coordinates,
                                    minConfidence: minConfidence,
                                    skipPersonClass: skipPersonClass)
    }

    // MARK: - 输出解析

    private static func parse(confidence: MLMultiArray,
                              coordinates: MLMultiArray,
                              minConfidence: Float,
                              skipPersonClass: Bool) -> [DetectedItem] {

        let confidenceShape = confidence.shape.map { $0.intValue }
        let coordinateShape = coordinates.shape.map { $0.intValue }

        guard confidenceShape.count == 2,
              coordinateShape.count == 2,
              coordinateShape[1] == 4 else {
            return []
        }

        let count = min(confidenceShape[0], coordinateShape[0])
        let classCount = confidenceShape[1]
        guard count > 0, classCount > 0 else { return [] }

        var items: [DetectedItem] = []
        items.reserveCapacity(count)

        for row in 0..<count {
            var bestClass = -1
            var bestScore: Float = 0
            for column in 0..<classCount {
                let score = value(confidence, row, column)
                if score > bestScore {
                    bestScore = score
                    bestClass = column
                }
            }

            guard bestClass >= 0, bestScore >= minConfidence else { continue }
            if skipPersonClass && bestClass == COCOLabels.personIndex { continue }

            // 归一化的 [cx, cy, w, h]
            let centerX = value(coordinates, row, 0)
            let centerY = value(coordinates, row, 1)
            let width = value(coordinates, row, 2)
            let height = value(coordinates, row, 3)

            guard width > 0.001, height > 0.001 else { continue }

            // 换算成 Vision 坐标系：原点左下，y 轴向上。
            // 模型给的是「原点左上、y 轴向下」的中心点坐标。
            let left = CGFloat(centerX - width / 2)
            let top = CGFloat(centerY - height / 2)
            let rect = CGRect(x: left,
                              y: 1 - top - CGFloat(height),
                              width: CGFloat(width),
                              height: CGFloat(height))

            let clipped = rect.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !clipped.isNull, clipped.width > 0.005, clipped.height > 0.005 else { continue }

            items.append(.object(label: COCOLabels.displayName(index: bestClass),
                                 confidence: bestScore,
                                 box: clipped))
        }

        return items
    }

    /// 读取二维多数组第 (row, column) 个元素，兼容 float32 / float16 / double。
    private static func value(_ array: MLMultiArray, _ row: Int, _ column: Int) -> Float {
        let strides = array.strides
        let offset = row * strides[0].intValue + column * strides[1].intValue

        switch array.dataType {
        case .float32:
            return array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)[offset]
        case .float16:
            return Float(array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)[offset])
        case .double:
            return Float(array.dataPointer.bindMemory(to: Double.self, capacity: array.count)[offset])
        default:
            return 0
        }
    }
}
