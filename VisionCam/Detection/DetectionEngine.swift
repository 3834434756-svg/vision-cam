import Combine
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import QuartzCore

/// 检测引擎：接收相机帧 → 跑物体检测 + 姿态检测 → 把结果发布到主线程供 UI 绘制。
///
/// 串行队列 + `busy` 标志做背压：上一帧还没跑完就丢弃新帧，
/// 保证 UI 不会被积压的推理任务拖垮。
final class DetectionEngine: ObservableObject {

    // MARK: 对外状态

    /// 当前帧的检测结果。
    @Published private(set) var items: [DetectedItem] = []
    /// 已按设备方向摆正的图像尺寸，用于坐标换算。
    @Published private(set) var orientedImageSize: CGSize = .zero
    /// 处理帧率。
    @Published private(set) var fps: Double = 0
    /// 单帧推理耗时（毫秒）。
    @Published private(set) var latencyMilliseconds: Double = 0
    /// 模型是否加载成功。
    @Published private(set) var isModelReady = false
    /// 加载失败原因。
    @Published private(set) var modelErrorMessage: String?

    // MARK: 用户可调

    /// 是否开启物体检测。
    @Published var detectObjects = true
    /// 是否开启人体姿态检测。
    @Published var detectPose = true
    /// 前端置信度门槛（模型内部门槛是 0.25，所以下限取 0.25）。
    @Published var minConfidence: Double = 0.35

    // MARK: 内部

    private let objectDetector: ObjectDetector?
    private let poseDetector = PoseDetector()

    private let queue = DispatchQueue(label: "com.visioncam.detection", qos: .userInitiated)
    private var isBusy = false
    private var lastProcessedTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0

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

        queue.async { [weak self] in
            guard let self = self else { return }
            guard !self.isBusy else { return }
            self.isBusy = true
            defer { self.isBusy = false }

            let start = CACurrentMediaTime()

            var results: [DetectedItem] = []

            if wantObjects, let detector = self.objectDetector {
                // 姿态检测开着时，人物交给姿态检测，避免同一个框画两遍
                results.append(contentsOf: detector.detect(pixelBuffer: pixelBuffer,
                                                           orientation: orientation,
                                                           minConfidence: threshold,
                                                           skipPersonClass: wantPose))
            }

            if wantPose {
                results.append(contentsOf: self.poseDetector.detect(pixelBuffer: pixelBuffer,
                                                                    orientation: orientation))
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

            let fps = self.smoothedFPS
            // 置信度高的排前面，画的时候压在下面
            let sorted = results.sorted { $0.confidence > $1.confidence }

            DispatchQueue.main.async {
                self.items = sorted
                self.orientedImageSize = orientedSize
                self.latencyMilliseconds = elapsed
                self.fps = fps
            }
        }
    }

    /// 清空当前结果（切后台 / 停相机时用）。
    func reset() {
        DispatchQueue.main.async {
            self.items = []
            self.fps = 0
            self.latencyMilliseconds = 0
        }
    }
}
