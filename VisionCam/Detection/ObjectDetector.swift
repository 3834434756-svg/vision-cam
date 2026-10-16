import CoreGraphics
import CoreML
import ImageIO
import Vision

/// 基于 YOLOv8n（CoreML）的物体检测器。
///
/// 模型是 Ultralytics 导出的 **带内置 NMS 的 CoreML pipeline**：
///   * 输入：`image`（640×640，RGB）
///   * 输出：`confidence` 形状 [N, 80]，`coordinates` 形状 [N, 4]
///
/// `coordinates` 每行是 `[cx, cy, w, h]`，**已归一化到 0~1**（Apple 官方
/// CoreML proto 对 pipeline NMS 的定义：x/y 是盒子中心点，w/h 是宽高）。
///
/// ## 关于裁剪方式（这是检测精度的关键）
///
/// 手机竖屏取景时画面是「高瘦」的（如 720×1280），而模型输入是正方形。
/// Vision 提供三种缩放方式，实测差异极大：
///
/// | 方式 | 是否失真 | 送进模型的内容 | 结果 |
/// |---|---|---|---|
/// | `.scaleFill` | 严重（宽高比 0.56 → 1.0） | 640×640 全图 | 小物体几乎检不出 |
/// | `.scaleFit` | 无 | 360×640（横向分辨率砍半） | 全画面但偏模糊 |
/// | `.centerCrop` | 无 | 640×640（仅中心正方形） | 最清晰 |
///
/// 这里选 **`.centerCrop`**：从画面中心裁出正方形再等比缩放到 640×640，
/// 既不失真也不损失分辨率，代价是上下边缘不在检测范围内。
/// 裁剪区域与预览画面中心对齐，所以只需把「裁剪正方形内的归一化坐标」
/// 映射回「整幅画面的归一化坐标」即可，映射关系见 `CropMapping`。
final class ObjectDetector {

    private static let modelName = "YOLOv8n"

    /// 复用同一个 request：`VNCoreMLRequest` 不是线程安全的，
    /// 但本类只在检测串行队列上使用，复用可以省掉每帧的构建开销。
    private let request: VNCoreMLRequest

    init?() {
        guard let model = ObjectDetector.loadModel(named: ObjectDetector.modelName),
              let vnModel = try? VNCoreMLModel(for: model) else {
            return nil
        }
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop
        self.request = request
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
    ///   - orientedSize: 摆正后的图像尺寸，用于把裁剪坐标映射回整幅画面。
    ///   - minConfidence: 前端置信度门槛（模型内部门槛是 0.25，这里只做二次过滤）。
    ///   - skipPersonClass: 是否跳过 `person` 类。姿态检测已单独处理人物时置 true，避免重复画框。
    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation,
                orientedSize: CGSize,
                minConfidence: Float,
                skipPersonClass: Bool) -> [DetectedItem] {

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
                                    skipPersonClass: skipPersonClass,
                                    crop: CropMapping(orientedSize: orientedSize))
    }

    // MARK: - 输出解析

    private static func parse(confidence: MLMultiArray,
                              coordinates: MLMultiArray,
                              minConfidence: Float,
                              skipPersonClass: Bool,
                              crop: CropMapping) -> [DetectedItem] {

        let confidenceShape = confidence.shape.map { $0.intValue }
        let coordinateShape = coordinates.shape.map { $0.intValue }

        guard confidenceShape.count >= 2,
              coordinateShape.count >= 2,
              coordinateShape.last == 4 else {
            return []
        }

        // 把 batch 维（若有）折算进行数
        let confidenceRows = confidenceShape.dropLast().reduce(1, *)
        let coordinateRows = coordinateShape.dropLast().reduce(1, *)
        let count = min(confidenceRows, coordinateRows)
        let classCount = confidenceShape[confidenceShape.count - 1]

        guard count > 0, classCount > 0 else { return [] }

        // 一次性取出裸指针与步长，避免在内层循环里反复重建 strides 数组
        let confidenceReader = MatrixReader(confidence)
        let coordinateReader = MatrixReader(coordinates)

        var items: [DetectedItem] = []
        items.reserveCapacity(min(count, 32))

        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)

        for row in 0..<count {
            var bestClass = -1
            var bestScore: Float = 0
            for column in 0..<classCount {
                let score = confidenceReader.value(row, column)
                if score > bestScore {
                    bestScore = score
                    bestClass = column
                }
            }

            guard bestClass >= 0, bestScore >= minConfidence else { continue }
            if skipPersonClass && bestClass == COCOLabels.personIndex { continue }

            // 模型给的 [cx, cy, w, h]：原点左上、y 向下，且相对「中心裁剪的正方形」
            let centerX = coordinateReader.value(row, 0)
            let centerY = coordinateReader.value(row, 1)
            let boxWidth = coordinateReader.value(row, 2)
            let boxHeight = coordinateReader.value(row, 3)

            guard boxWidth > 0.001, boxHeight > 0.001 else { continue }

            let leftInCrop = CGFloat(centerX - boxWidth / 2)
            let topInCrop = CGFloat(centerY - boxHeight / 2)

            // 映射回整幅画面（仍是 y 向下）
            let fullLeft = crop.originX + leftInCrop * crop.scaleX
            let fullTop = crop.originY + topInCrop * crop.scaleY
            let fullWidth = CGFloat(boxWidth) * crop.scaleX
            let fullHeight = CGFloat(boxHeight) * crop.scaleY

            // 换成 Vision 坐标：原点左下、y 轴向上
            let rect = CGRect(x: fullLeft,
                              y: 1 - fullTop - fullHeight,
                              width: fullWidth,
                              height: fullHeight)

            let clipped = rect.intersection(unit)
            guard !clipped.isNull, clipped.width > 0.005, clipped.height > 0.005 else { continue }

            let label = COCOLabels.displayName(index: bestClass)
            if bestClass == COCOLabels.personIndex {
                // 姿态检测关闭时人物由 YOLO 负责，走人物配色（无骨架）
                items.append(.person(label: label,
                                     confidence: bestScore,
                                     box: clipped,
                                     joints: [:],
                                     jointConfidences: [:]))
            } else {
                items.append(.object(label: label, confidence: bestScore, box: clipped))
            }
        }

        return items
    }
}

// MARK: - 中心裁剪的坐标换算

/// 描述「中心裁剪出的正方形」与「整幅摆正画面」之间的位置关系。
///
/// 模型看到的是裁剪后的正方形，所以它吐出的归一化坐标是相对该正方形的；
/// 而预览显示的是整幅画面，画框前必须先映射回去。
///
/// 裁剪是居中的，因此 y 向下与 y 向上两套坐标系下的原点偏移量相同。
private struct CropMapping {

    /// 裁剪正方形左上角在整幅画面中的归一化位置。
    let originX: CGFloat
    let originY: CGFloat
    /// 裁剪正方形边长占整幅画面宽 / 高的比例。
    let scaleX: CGFloat
    let scaleY: CGFloat

    init(orientedSize: CGSize) {
        let width = max(orientedSize.width, 1)
        let height = max(orientedSize.height, 1)
        let side = min(width, height)
        let scaleX = side / width
        let scaleY = side / height

        self.originX = (1 - scaleX) / 2
        self.originY = (1 - scaleY) / 2
        self.scaleX = scaleX
        self.scaleY = scaleY
    }
}

// MARK: - 多数组读取

/// 轻量二维读取器：提前缓存裸指针与步长，兼容 float32 / float16 / double。
///
/// 直接照搬 `MLMultiArray` 的 strides 读取，因此无论数组是否连续、
/// 是否带 batch 维都能正确取值。
private struct MatrixReader {

    private let pointer: UnsafeMutableRawPointer
    private let rowStride: Int
    private let columnStride: Int
    private let dataType: MLMultiArrayDataType

    init(_ array: MLMultiArray) {
        self.pointer = array.dataPointer
        self.dataType = array.dataType

        let strides = array.strides.map { $0.intValue }
        if array.shape.count >= 3, strides.count >= 3 {
            // 带 batch 维（恒为 1）：取最后两维的步长
            self.rowStride = strides[strides.count - 2]
            self.columnStride = strides[strides.count - 1]
        } else if strides.count >= 2 {
            self.rowStride = strides[0]
            self.columnStride = strides[1]
        } else {
            self.rowStride = 0
            self.columnStride = strides.first ?? 1
        }
    }

    func value(_ row: Int, _ column: Int) -> Float {
        let elementOffset = row * rowStride + column * columnStride
        switch dataType {
        case .float32:
            return pointer.load(fromByteOffset: elementOffset * MemoryLayout<Float32>.size,
                                as: Float32.self)
        case .float16:
            return Float(pointer.load(fromByteOffset: elementOffset * MemoryLayout<Float16>.size,
                                      as: Float16.self))
        case .double:
            return Float(pointer.load(fromByteOffset: elementOffset * MemoryLayout<Double>.size,
                                      as: Double.self))
        default:
            return 0
        }
    }
}
