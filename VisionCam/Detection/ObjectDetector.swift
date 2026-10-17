import CoreGraphics
import CoreML
import Foundation
import ImageIO
import Vision

/// 基于 YOLO11s（CoreML）的物体检测器。
///
/// 模型是 Ultralytics 导出的 **带内置 NMS 的 CoreML pipeline**：
///   * 输入：`image`（640×640，RGB）
///   * 输出：`confidence` 形状 [N, 80]，`coordinates` 形状 [N, 4]
///
/// `coordinates` 每行是 `[cx, cy, w, h]`，**已归一化到 0~1**（Apple 官方
/// CoreML proto 对 pipeline NMS 的定义：x/y 是盒子中心点，w/h 是宽高）。
///
/// ## 为什么不再只用「中心裁剪」
///
/// 手机竖屏取景时画面是「高瘦」的（720×1280），而模型输入是正方形 640×640。
/// 早期版本为了不损失分辨率，只把画面**正中间的正方形**送进模型——
/// 这等于把上下各 22% 的画面直接丢掉，检测范围只有整幅画面的 56%。
/// 人物由姿态检测负责（跑的是整幅画面），所以「人一直有框、桌上的杯子书本没有框」
/// 就成了必然结果，而不是模型不行。
///
/// 现在改成 **分块扫描（tiling）**：把整幅画面切成若干互相重叠的正方形瓦片，
/// 每帧只跑其中一块、轮流刷新，再把各块结果按 IoU 合并。这样
///   * 覆盖范围 = 100%（不再有盲区）；
///   * 每一块都是「原分辨率 → 640」的等比缩放，小物体不会被缩小两次；
///   * 每帧仍然只跑一次模型，耗时与之前持平。
final class ObjectDetector {

    private static let modelName = "YOLO11s"

    /// 一次检测的产出，除了结果本身还带上诊断信息。
    ///
    /// 之前如果 Vision 返回的结构和预期不符，解析会静默返回空数组，
    /// 表现就是「只有姿态检测的人物有框」。现在把失败原因带出去，UI 能直接提示。
    struct Outcome {
        var items: [DetectedItem] = []
        /// 模型 NMS 输出的候选框数量（过滤前）。
        var rawBoxCount: Int = 0
        /// 非 nil 表示这一帧解析失败，内容为原因。
        var failure: String?
    }

    /// 一块检测区域：整幅画面被切成的正方形之一。
    struct Tile {
        /// Vision 的 `regionOfInterest`（原点左下、归一化）。
        let roi: CGRect
        /// 同一块区域在「原点左上」归一化坐标下的位置，用于把模型输出映射回整幅画面。
        let rect: CGRect
    }

    /// 复用同一个 request：`VNCoreMLRequest` 不是线程安全的，
    /// 但本类只在检测串行队列上使用，复用可以省掉每帧的构建开销。
    private let request: VNCoreMLRequest

    init?() {
        guard let model = ObjectDetector.loadModel(named: ObjectDetector.modelName),
              let vnModel = try? VNCoreMLModel(for: model) else {
            return nil
        }
        let request = VNCoreMLRequest(model: vnModel)
        // 送进来的区域已经是正方形，scaleFill 不会造成任何形变
        request.imageCropAndScaleOption = .scaleFill
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

    // MARK: - 分块

    /// 把整幅画面切成覆盖全部区域、互相重叠的正方形瓦片。
    ///
    /// 竖屏（高 > 宽）沿纵向切，横屏沿横向切；正方形画面只有一块。
    /// 重叠比例取 22%，保证跨越边界的物体至少在某一块里是完整的。
    static func tiles(for orientedSize: CGSize) -> [Tile] {
        let width = max(orientedSize.width, 1)
        let height = max(orientedSize.height, 1)
        let side = min(width, height)

        let overlapRatio: CGFloat = 0.22
        let step = max(side * (1 - overlapRatio), 1)

        var tiles: [Tile] = []

        if height >= width {
            let span = height - side
            let count = span <= 1 ? 1 : Int(ceil(span / step)) + 1
            for index in 0..<count {
                let top = count == 1 ? span / 2 : span * CGFloat(index) / CGFloat(count - 1)
                tiles.append(Tile(
                    roi: CGRect(x: 0,
                                y: 1 - (top + side) / height,
                                width: 1,
                                height: side / height),
                    rect: CGRect(x: 0,
                                 y: top / height,
                                 width: 1,
                                 height: side / height)
                ))
            }
        } else {
            let span = width - side
            let count = span <= 1 ? 1 : Int(ceil(span / step)) + 1
            for index in 0..<count {
                let left = count == 1 ? span / 2 : span * CGFloat(index) / CGFloat(count - 1)
                tiles.append(Tile(
                    roi: CGRect(x: left / width,
                                y: 0,
                                width: side / width,
                                height: 1),
                    rect: CGRect(x: left / width,
                                 y: 0,
                                 width: side / width,
                                 height: 1)
                ))
            }
        }

        return tiles
    }

    // MARK: - 检测

    /// 在给定帧的指定瓦片上跑一次检测。
    ///
    /// - Parameters:
    ///   - pixelBuffer: 相机输出帧。
    ///   - orientation: 把该帧摆正所需的 EXIF 方向。
    ///   - tile: 本次要扫描的区域。
    ///   - minConfidence: 前端置信度门槛（模型内部门槛已调到 0.10，这里只做二次过滤）。
    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation,
                tile: Tile,
                minConfidence: Float) -> Outcome {

        request.regionOfInterest = tile.roi

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            return Outcome(failure: "推理失败：\(error.localizedDescription)")
        }

        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            return Outcome(failure: "模型输出类型异常")
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
            return Outcome(failure: "缺少 confidence / coordinates 输出")
        }

        return parse(confidence: confidence,
                     coordinates: coordinates,
                     mapping: tile.rect,
                     minConfidence: minConfidence)
    }

    // MARK: - 输出解析

    private func parse(confidence: MLMultiArray,
                       coordinates: MLMultiArray,
                       mapping: CGRect,
                       minConfidence: Float) -> Outcome {

        guard let layout = Layout(confidence: confidence, coordinates: coordinates) else {
            return Outcome(failure: "输出张量形状异常")
        }

        var items: [DetectedItem] = []
        items.reserveCapacity(min(layout.boxCount, 32))

        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)

        for row in 0..<layout.boxCount {
            var bestClass = -1
            var bestScore: Float = 0
            for column in 0..<layout.classCount {
                let score = layout.confidence.value(row, column)
                if score > bestScore {
                    bestScore = score
                    bestClass = column
                }
            }

            guard bestClass >= 0, bestScore >= minConfidence else { continue }

            // 模型给的 [cx, cy, w, h]：原点左上、y 向下，且相对「本块瓦片」
            let centerX = CGFloat(layout.coordinates.value(row, 0))
            let centerY = CGFloat(layout.coordinates.value(row, 1))
            let boxWidth = CGFloat(layout.coordinates.value(row, 2))
            let boxHeight = CGFloat(layout.coordinates.value(row, 3))

            guard boxWidth > 0.001, boxHeight > 0.001 else { continue }

            // 映射回整幅画面（仍是 y 向下）
            let fullLeft = mapping.minX + (centerX - boxWidth / 2) * mapping.width
            let fullTop = mapping.minY + (centerY - boxHeight / 2) * mapping.height
            let fullWidth = boxWidth * mapping.width
            let fullHeight = boxHeight * mapping.height

            // 换成 Vision 坐标：原点左下、y 轴向上
            let rect = CGRect(x: fullLeft,
                              y: 1 - fullTop - fullHeight,
                              width: fullWidth,
                              height: fullHeight)

            let clipped = rect.intersection(unit)
            guard !clipped.isNull, clipped.width > 0.005, clipped.height > 0.005 else { continue }

            let label = COCOLabels.displayName(index: bestClass)
            if bestClass == COCOLabels.personIndex {
                // 没有骨架的「人物」，由引擎和姿态检测的结果合并
                items.append(.person(label: label,
                                     classIndex: bestClass,
                                     confidence: bestScore,
                                     box: clipped,
                                     joints: [:],
                                     jointConfidences: [:]))
            } else {
                items.append(.object(label: label,
                                     classIndex: bestClass,
                                     confidence: bestScore,
                                     box: clipped))
            }
        }

        return Outcome(items: items,
                       rawBoxCount: layout.boxCount,
                       failure: nil)
    }

    // MARK: - 跨瓦片去重

    /// 同一类别的框做贪心 NMS，用来合并相邻瓦片里重复检出的同一物体。
    ///
    /// 跨块边界的物体会在相邻两块里各出现一次，而且**各被切掉一截**。
    /// 这时如果只保留置信度高的那个，框就会在接缝处断掉。所以当两个框
    /// 大部分互相包含（交集占较小框面积 80% 以上）时，直接取并集，
    /// 拼回完整的目标；否则按普通 NMS 丢弃低分框。
    static func suppressDuplicates(_ items: [DetectedItem],
                                   iouThreshold: CGFloat = 0.45) -> [DetectedItem] {
        guard items.count > 1 else { return items }

        var kept: [DetectedItem] = []
        kept.reserveCapacity(items.count)

        for item in items.sorted(by: { $0.confidence > $1.confidence }) {
            var merged = false

            for index in kept.indices where kept[index].classIndex == item.classIndex {
                let other = kept[index]
                guard BoxMath.iou(other.box, item.box) > iouThreshold else { continue }

                let intersection = other.box.intersection(item.box)
                let intersectionArea = intersection.isNull ? 0 : intersection.width * intersection.height
                let minArea = min(other.box.width * other.box.height,
                                  item.box.width * item.box.height)
                let containment = minArea > 0 ? intersectionArea / minArea : 0

                if containment > 0.8 {
                    // 同一物体被瓦片边界切开的两半，拼回完整框
                    kept[index] = other.replacing(box: other.box.union(item.box))
                }
                merged = true
                break
            }

            if !merged {
                kept.append(item)
            }
        }

        return kept
    }
}

// MARK: - 输出张量布局

/// 描述 NMS 输出的二维布局，兼容两种常见排布：
///   * 框优先：`[N, C]` / `[1, N, C]`
///   * 通道优先：`[C, N]` / `[1, C, N]`
///
/// 理论上本模型是前者，但不同 coremltools 版本导出结果可能不同，
/// 这里两种都接住，避免「解析不出来就静默返回空数组」。
private struct Layout {

    let confidence: MatrixReader
    let coordinates: MatrixReader
    let boxCount: Int
    let classCount: Int

    init?(confidence: MLMultiArray, coordinates: MLMultiArray) {
        let confidenceShape = confidence.shape.map { $0.intValue }
        let coordinateShape = coordinates.shape.map { $0.intValue }

        guard !confidenceShape.isEmpty, !coordinateShape.isEmpty else { return nil }

        let boxMajor: Bool
        let boxCount: Int
        let classCount: Int

        if coordinateShape.last == 4 {
            // [N, 4] 或 [1, N, 4]
            boxMajor = true
            boxCount = coordinateShape.dropLast().reduce(1, *)
            // 单类模型的 confidence 可能是 [N]（每行只有一个分数）
            classCount = confidenceShape.count == 1 ? 1 : (confidenceShape.last ?? 1)
        } else if coordinateShape.count >= 2,
                  coordinateShape[coordinateShape.count - 2] == 4 {
            // [4, N] 或 [1, 4, N]
            boxMajor = false
            boxCount = coordinateShape.last ?? 0
            classCount = confidenceShape.count >= 2
                ? confidenceShape[confidenceShape.count - 2]
                : (confidenceShape.last ?? 1)
        } else {
            return nil
        }

        // boxCount 允许为 0：NMS 一帧里什么都没检出来是正常情况，不是错误
        guard classCount > 0 else { return nil }

        self.boxCount = boxCount
        self.classCount = classCount
        self.confidence = MatrixReader(confidence, boxMajor: boxMajor)
        self.coordinates = MatrixReader(coordinates, boxMajor: boxMajor)
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

    init(_ array: MLMultiArray, boxMajor: Bool) {
        self.pointer = array.dataPointer
        self.dataType = array.dataType

        let strides = array.strides.map { $0.intValue }

        if strides.count < 2 {
            // 单类模型可能输出一维 [N]
            self.rowStride = strides.first ?? 1
            self.columnStride = 0
            return
        }

        if boxMajor {
            self.rowStride = strides[strides.count - 2]
            self.columnStride = strides[strides.count - 1]
        } else {
            self.rowStride = strides[strides.count - 1]
            self.columnStride = strides[strides.count - 2]
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
