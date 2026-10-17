import Combine
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import QuartzCore

/// 检测引擎：接收相机帧 → 跑物体检测 + 姿态检测 → 把结果发布到主线程供 UI 绘制。
///
/// ## 调度策略
///
/// 1. **串行队列 + `busy` 背压**：上一帧还没跑完就丢弃新帧，保证 UI 不被积压的推理拖垮。
/// 2. **物体检测：每帧一次，瓦片轮转**。整幅画面被切成若干正方形瓦片（竖屏时通常 2 块，
///    覆盖 100% 画面），每帧只跑其中一块，各块结果各自缓存后按 IoU 合并。
///    每块约 15fps 刷新，合并后整体仍跟随相机帧率。
/// 3. **姿态检测：每两帧一次**。姿态本来就比物体检测吃时间，降到 15fps 肉眼几乎无感，
///    省下的时间留给物体检测，保证每帧都有新的瓦片结果。
///
/// 这样平均每帧约 1.5 次推理（之前交替执行是 1 次），换来的是物体检测
/// 覆盖范围从画面中间的 56% 变成 100%。如果机型偏慢，关掉姿态骨架即可回到 1 次。
final class DetectionEngine: ObservableObject {

    // MARK: 对外状态

    /// 当前帧的检测结果。
    @Published private(set) var items: [DetectedItem] = []
    /// 已按设备方向摆正的图像尺寸，用于坐标换算。
    @Published private(set) var orientedImageSize: CGSize = .zero
    /// 处理帧率。
    @Published private(set) var fps: Double = 0
    /// 单帧推理耗时（毫秒，平滑后）。
    @Published private(set) var latencyMilliseconds: Double = 0
    /// 模型是否加载成功。
    @Published private(set) var isModelReady = false
    /// 加载失败原因。
    @Published private(set) var modelErrorMessage: String?
    /// 当前画面的分块数量（1 表示画面本身是正方形）。
    @Published private(set) var tileCount = 1
    /// 上一帧模型 NMS 给出的候选框数量（过滤前），用于判断模型是否真的在工作。
    @Published private(set) var rawCandidateCount = 0
    /// 物体检测链路异常时的提示，nil 表示正常。
    @Published private(set) var diagnosticMessage: String?

    // MARK: 用户可调

    /// 是否开启物体检测。
    @Published var detectObjects = true
    /// 是否开启人体姿态检测。
    @Published var detectPose = true
    /// 前端置信度门槛（模型内部门槛是 0.10，所以下限取 0.10）。
    @Published var minConfidence: Double = 0.25

    // MARK: 内部

    private let objectDetector: ObjectDetector?
    private let poseDetector = PoseDetector()

    private let queue = DispatchQueue(label: "com.visioncam.detection", qos: .userInitiated)

    /// 真正的背压开关：在相机回调线程上同步读写。
    ///
    /// 注意不能把「是否在忙」放在串行队列内部判断 —— 串行队列一次只跑一个 block，
    /// 进去时永远是空闲的，判断恒为假，帧只会无限堆积，画面越跑越滞后。
    private let gate = NSLock()
    private var isFrameInFlight = false

    private var lastProcessedTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0
    private var smoothedLatency: Double = 0

    /// 帧计数器。
    private var frameIndex: UInt64 = 0

    /// 当前的分块方案。
    private var tiles: [ObjectDetector.Tile] = []
    private var tiledSize: CGSize = .zero
    /// 每块瓦片最近一次的结果与时间戳。
    private var tileItems: [[DetectedItem]] = []
    private var tileStamps: [CFTimeInterval] = []
    private var nextTile = 0

    /// 姿态结果与时间戳。
    private var cachedPoses: [DetectedItem] = []
    private var poseStamp: CFTimeInterval = 0

    /// 上一帧的结果，用于方框平滑。
    private var smoothedItems: [DetectedItem] = []

    /// 只在检测队列上读写，主线程发布时再拷贝出去。
    private var latestCandidateCount = 0
    private var latestDiagnostic: String?

    /// 缓存有效期：超过这个时间的结果不再参与合并，避免留下「鬼影」。
    private static let cacheLifetime: CFTimeInterval = 0.35

    init() {
        let detector = ObjectDetector()
        self.objectDetector = detector
        self.isModelReady = detector != nil
        if detector == nil {
            self.modelErrorMessage = "未能加载 YOLO11s 模型，物体识别不可用。"
        }
    }

    /// 相机每来一帧调用一次（在相机的输出队列上）。
    func process(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        // 旋转 90° 的方向要把宽高对调，才是「摆正后」的图像尺寸
        let swapsDimensions: Bool
        switch orientation {
        case .left, .right, .leftMirrored, .rightMirrored:
            swapsDimensions = true
        default:
            swapsDimensions = false
        }
        let orientedSize = swapsDimensions
            ? CGSize(width: height, height: width)
            : CGSize(width: width, height: height)

        // 在主线程之外读取开关，避免每帧跨线程取属性
        let wantObjects = detectObjects
        let wantPose = detectPose
        let threshold = Float(minConfidence)

        // 真正的丢帧：上一帧还在推理就直接丢掉这一帧，避免队列无限堆积
        gate.lock()
        if isFrameInFlight {
            gate.unlock()
            return
        }
        isFrameInFlight = true
        gate.unlock()

        queue.async { [weak self] in
            guard let self = self else { return }
            defer {
                self.gate.lock()
                self.isFrameInFlight = false
                self.gate.unlock()
            }

            let start = CACurrentMediaTime()
            self.frameIndex &+= 1
            self.prepareTiles(for: orientedSize)

            // 姿态每两帧跑一次，省下的时间给物体检测
            let runPose = wantPose && (self.frameIndex % 2 == 0)

            if wantObjects, let detector = self.objectDetector, !self.tiles.isEmpty {
                let index = self.nextTile % self.tiles.count
                let outcome = detector.detect(pixelBuffer: pixelBuffer,
                                              orientation: orientation,
                                              tile: self.tiles[index],
                                              minConfidence: threshold)
                self.tileItems[index] = outcome.items
                self.tileStamps[index] = CACurrentMediaTime()
                self.nextTile = (index + 1) % self.tiles.count
                self.latestCandidateCount = outcome.rawBoxCount
                self.latestDiagnostic = outcome.failure
            } else if !wantObjects {
                self.latestCandidateCount = 0
                self.latestDiagnostic = nil
            }

            if runPose {
                self.cachedPoses = self.poseDetector.detect(pixelBuffer: pixelBuffer,
                                                            orientation: orientation)
                self.poseStamp = CACurrentMediaTime()
            }

            let elapsed = (CACurrentMediaTime() - start) * 1000

            let now = CACurrentMediaTime()
            if self.lastProcessedTime > 0 {
                let delta = now - self.lastProcessedTime
                if delta > 0.0001 {
                    let instant = 1.0 / delta
                    self.smoothedFPS = self.smoothedFPS == 0
                        ? instant
                        : self.smoothedFPS * 0.85 + instant * 0.15
                }
            }
            self.lastProcessedTime = now

            // 跳过前几帧：首帧包含模型预热，耗时没有参考价值
            if self.frameIndex > 3 {
                self.smoothedLatency = self.smoothedLatency == 0
                    ? elapsed
                    : self.smoothedLatency * 0.8 + elapsed * 0.2
            }

            // 合并各瓦片的结果（跨块去重）
            var objects: [DetectedItem] = []
            if wantObjects {
                for index in self.tiles.indices where now - self.tileStamps[index] < DetectionEngine.cacheLifetime {
                    objects.append(contentsOf: self.tileItems[index])
                }
                objects = ObjectDetector.suppressDuplicates(objects, iouThreshold: 0.45)
            }

            // 姿态结果也带有效期，避免关了再开时残留旧骨架
            var poses: [DetectedItem] = []
            if wantPose, now - self.poseStamp < DetectionEngine.cacheLifetime {
                poses = self.cachedPoses
            }

            // 注意：置信度门槛在物体检测内部已经过滤过；姿态结果不受滑杆影响，
            // 否则把门槛调高会把「人物」一起筛掉。
            let fused = DetectionEngine.fusePersons(poses: poses, objects: objects)
            var results = fused.objects + fused.persons
            // 置信度高的排前面，画的时候压在下面
            results.sort { $0.confidence > $1.confidence }

            let smoothed = self.smooth(results)

            let fps = self.smoothedFPS
            let latency = self.smoothedLatency
            let tileCount = self.tiles.count
            let candidates = self.latestCandidateCount
            let diagnostic = self.latestDiagnostic

            DispatchQueue.main.async {
                self.items = smoothed
                self.orientedImageSize = orientedSize
                self.latencyMilliseconds = latency
                self.fps = fps
                self.tileCount = max(tileCount, 1)
                self.rawCandidateCount = candidates
                self.diagnosticMessage = diagnostic
            }
        }
    }

    // MARK: - 分块

    /// 画面尺寸变化时重建分块方案。
    private func prepareTiles(for size: CGSize) {
        guard size != tiledSize else { return }

        tiledSize = size
        tiles = ObjectDetector.tiles(for: size)
        tileItems = Array(repeating: [], count: tiles.count)
        tileStamps = Array(repeating: 0, count: tiles.count)
        nextTile = 0
        smoothedItems = []
    }

    // MARK: - 人物合并

    /// 把姿态检测和 YOLO 的 person 类合并，保证「一个人只有一个框」。
    ///
    /// 姿态的框来自关键点外接框，会随手臂摆动而跳动；YOLO 的框更稳。
    /// 所以两者匹配上时用 YOLO 的框、保留骨架；姿态漏检时仍然保留 YOLO 的框，
    /// 不会出现「人在画面里但没有框」。
    private static func fusePersons(poses: [DetectedItem],
                                    objects: [DetectedItem]) -> (objects: [DetectedItem], persons: [DetectedItem]) {

        let others = objects.filter { $0.classIndex != COCOLabels.personIndex }
        let yoloPersons = objects.filter { $0.classIndex == COCOLabels.personIndex }

        guard !poses.isEmpty else {
            return (others, yoloPersons)
        }

        var fused: [DetectedItem] = []
        fused.reserveCapacity(max(poses.count, yoloPersons.count))
        var matched = Set<Int>()

        for pose in poses {
            var bestIndex: Int?
            var bestIoU: CGFloat = 0.2

            for (index, person) in yoloPersons.enumerated() where !matched.contains(index) {
                let score = BoxMath.iou(pose.box, person.box)
                if score > bestIoU {
                    bestIoU = score
                    bestIndex = index
                }
            }

            if let index = bestIndex {
                matched.insert(index)
                let yoloBox = yoloPersons[index].box
                let poseArea = pose.box.width * pose.box.height
                let yoloArea = yoloBox.width * yoloBox.height
                // YOLO 的框更稳，但可能被瓦片边界切掉一截；明显偏小时仍用姿态的框
                let box = yoloArea >= poseArea * 0.75 ? yoloBox : pose.box
                fused.append(pose.replacing(box: box,
                                            confidence: max(pose.confidence, yoloPersons[index].confidence)))
            } else {
                fused.append(pose)
            }
        }

        for (index, person) in yoloPersons.enumerated() where !matched.contains(index) {
            fused.append(person)
        }

        return (others, fused)
    }

    // MARK: - 平滑

    /// 按类别 + IoU 匹配上一帧的框，做指数平滑，减少抖动。
    private func smooth(_ current: [DetectedItem]) -> [DetectedItem] {
        guard !smoothedItems.isEmpty else {
            smoothedItems = current
            return current
        }

        let previous = smoothedItems
        var used = Set<Int>()
        var result: [DetectedItem] = []
        result.reserveCapacity(current.count)

        for item in current {
            var bestIndex: Int?
            var bestIoU: CGFloat = 0.25

            for (index, old) in previous.enumerated() where !used.contains(index) {
                guard old.classIndex == item.classIndex else { continue }
                let score = BoxMath.iou(old.box, item.box)
                if score > bestIoU {
                    bestIoU = score
                    bestIndex = index
                }
            }

            if let index = bestIndex {
                used.insert(index)
                let old = previous[index]
                result.append(item.replacing(
                    box: BoxMath.lerp(old.box, item.box, t: 0.6),
                    confidence: item.confidence * 0.6 + old.confidence * 0.4
                ))
            } else {
                result.append(item)
            }
        }

        smoothedItems = result
        return result
    }

    /// 清空当前结果（切后台 / 停相机时用）。
    func reset() {
        queue.async { [weak self] in
            guard let self = self else { return }
            self.tileItems = Array(repeating: [], count: self.tiles.count)
            self.tileStamps = Array(repeating: 0, count: self.tiles.count)
            self.nextTile = 0
            self.cachedPoses = []
            self.poseStamp = 0
            self.smoothedItems = []
            self.frameIndex = 0
            self.smoothedFPS = 0
            self.smoothedLatency = 0
            self.lastProcessedTime = 0
            self.latestCandidateCount = 0
            self.latestDiagnostic = nil
            DispatchQueue.main.async {
                self.items = []
                self.fps = 0
                self.latencyMilliseconds = 0
                self.rawCandidateCount = 0
                self.diagnosticMessage = nil
            }
        }
    }
}
