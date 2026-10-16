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
/// 2. **交替执行**：物体检测和姿态检测都开着时，每帧只跑其中一个（奇数帧跑物体、
///    偶数帧跑姿态），两个结果各自缓存、合并后一起发布。这样单帧耗时几乎减半，
///    整体帧率翻倍；代价是每个检测器的结果刷新率约为相机帧率的一半，
///    对 30fps 的相机来说仍有约 15fps，肉眼可接受。
///    只开一个检测器时该检测器每帧都跑，不降频。
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
    /// 物体检测实际覆盖的区域（Vision 归一化坐标，原点左下）。
    ///
    /// 物体检测走中心裁剪，上下边缘落在检测范围之外；UI 用它画取景提示，
    /// 让用户知道把目标放在哪里最准。物体检测关闭时为 nil。
    @Published private(set) var objectRegion: CGRect?

    // MARK: 用户可调

    /// 是否开启物体检测。
    @Published var detectObjects = true
    /// 是否开启人体姿态检测。
    @Published var detectPose = true
    /// 前端置信度门槛（模型内部门槛是 0.25，所以下限取 0.25）。
    @Published var minConfidence: Double = 0.30

    // MARK: 内部

    private let objectDetector: ObjectDetector?
    private let poseDetector = PoseDetector()

    private let queue = DispatchQueue(label: "com.visioncam.detection", qos: .userInitiated)
    private var isBusy = false
    private var lastProcessedTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0
    private var smoothedLatency: Double = 0

    /// 帧计数器，用于让两个检测器交替执行。
    private var frameIndex: UInt64 = 0
    /// 交替执行时，另一半的检测结果沿用上一轮，合并后一起发布。
    private var cachedObjects: [DetectedItem] = []
    private var cachedPoses: [DetectedItem] = []

    init() {
        let detector = ObjectDetector()
        self.objectDetector = detector
        self.isModelReady = detector != nil
        if detector == nil {
            self.modelErrorMessage = "未能加载 YOLOv8n 模型，物体识别不可用。"
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
        let region = wantObjects ? DetectionEngine.objectRegion(for: orientedSize) : nil

        queue.async { [weak self] in
            guard let self = self else { return }
            guard !self.isBusy else { return }
            self.isBusy = true
            defer { self.isBusy = false }

            let start = CACurrentMediaTime()
            self.frameIndex &+= 1

            let runObjects: Bool
            let runPose: Bool
            if wantObjects && wantPose {
                // 两个都开 → 交替，每帧只跑一个模型
                let objectsTurn = self.frameIndex % 2 == 1
                runObjects = objectsTurn
                runPose = !objectsTurn
            } else {
                runObjects = wantObjects
                runPose = wantPose
            }

            if runObjects, let detector = self.objectDetector {
                // 姿态检测开着时，人物交给姿态检测，避免同一个框画两遍
                self.cachedObjects = detector.detect(pixelBuffer: pixelBuffer,
                                                     orientation: orientation,
                                                     orientedSize: orientedSize,
                                                     minConfidence: threshold,
                                                     skipPersonClass: wantPose)
            }

            if runPose {
                self.cachedPoses = self.poseDetector.detect(pixelBuffer: pixelBuffer,
                                                            orientation: orientation)
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

            // 开关关掉时立刻清空对应结果，避免残留旧框
            var results: [DetectedItem] = []
            if wantObjects { results.append(contentsOf: self.cachedObjects) }
            if wantPose { results.append(contentsOf: self.cachedPoses) }

            let fps = self.smoothedFPS
            let latency = self.smoothedLatency
            // 置信度高的排前面，画的时候压在下面
            let sorted = results.sorted { $0.confidence > $1.confidence }

            DispatchQueue.main.async {
                self.items = sorted
                self.orientedImageSize = orientedSize
                self.objectRegion = region
                self.latencyMilliseconds = latency
                self.fps = fps
            }
        }
    }

    /// 物体检测覆盖的归一化区域：即中心裁剪出的正方形在整幅画面中的位置。
    ///
    /// 裁剪是居中的，所以 y 向下与 y 向上两套坐标系下的偏移量相同。
    private static func objectRegion(for orientedSize: CGSize) -> CGRect {
        let width = max(orientedSize.width, 1)
        let height = max(orientedSize.height, 1)
        let side = min(width, height)
        let normalizedWidth = side / width
        let normalizedHeight = side / height
        return CGRect(x: (1 - normalizedWidth) / 2,
                      y: (1 - normalizedHeight) / 2,
                      width: normalizedWidth,
                      height: normalizedHeight)
    }

    /// 清空当前结果（切后台 / 停相机时用）。
    func reset() {
        queue.async { [weak self] in
            guard let self = self else { return }
            self.cachedObjects = []
            self.cachedPoses = []
            self.frameIndex = 0
            self.smoothedFPS = 0
            self.smoothedLatency = 0
            self.lastProcessedTime = 0
            DispatchQueue.main.async {
                self.items = []
                self.fps = 0
                self.latencyMilliseconds = 0
            }
        }
    }
}
