import AVFoundation
import Combine
import ImageIO
import UIKit

/// 相机管理：负责 AVCaptureSession 的配置、权限申请、前后摄像头切换，
/// 并把每一帧原始像素缓冲连同「摆正它所需的 EXIF 方向」回调出去。
///
/// 工程锁定竖屏，因此方向映射是固定的：
///   * 后置摄像头 → `.right`
///   * 前置摄像头 → `.leftMirrored`
/// 这也和预览层的默认显示（竖屏、前置自动镜像）保持一致，
/// 所以画出来的框和骨架不会和画面错位。
final class CameraManager: NSObject, ObservableObject {

    let session = AVCaptureSession()

    @Published private(set) var isRunning = false
    @Published private(set) var authorizationDenied = false
    @Published private(set) var usingFrontCamera = false

    /// 每来一帧回调一次，在相机输出队列上执行。
    var onFrame: ((CVPixelBuffer, CGImagePropertyOrientation) -> Void)?

    private let sessionQueue = DispatchQueue(label: "com.visioncam.camera.session")
    private let outputQueue = DispatchQueue(label: "com.visioncam.camera.output", qos: .userInitiated)

    private var videoOutput: AVCaptureVideoDataOutput?
    private var currentInput: AVCaptureDeviceInput?
    private var isConfigured = false

    // MARK: - 生命周期

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureIfNeeded()
            startSession()

        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self = self else { return }
                if granted {
                    self.configureIfNeeded()
                    self.startSession()
                } else {
                    DispatchQueue.main.async { self.authorizationDenied = true }
                }
            }

        default:
            DispatchQueue.main.async { self.authorizationDenied = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    // MARK: - 切换摄像头

    func switchCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self, let oldInput = self.currentInput else { return }

            let newPosition: AVCaptureDevice.Position = oldInput.device.position == .back ? .front : .back
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video,
                                                       position: newPosition),
                  let newInput = try? AVCaptureDeviceInput(device: device) else { return }

            self.session.beginConfiguration()
            self.session.removeInput(oldInput)
            if self.session.canAddInput(newInput) {
                self.session.addInput(newInput)
                self.currentInput = newInput
            } else {
                self.session.addInput(oldInput)
            }
            self.session.commitConfiguration()

            let isFront = self.currentInput?.device.position == .front
            DispatchQueue.main.async { self.usingFrontCamera = isFront }
        }
    }

    // MARK: - 配置

    private func configureIfNeeded() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard !self.isConfigured else { return }
            self.isConfigured = true

            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1280x720

            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
               let input = try? AVCaptureDeviceInput(device: device),
               self.session.canAddInput(input) {
                self.session.addInput(input)
                self.currentInput = input
            }

            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            ]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: self.outputQueue)

            if self.session.canAddOutput(output) {
                self.session.addOutput(output)
                self.videoOutput = output
            }

            self.session.commitConfiguration()
        }
    }

    private func startSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            let running = self.session.isRunning
            DispatchQueue.main.async { self.isRunning = running }
        }
    }
}

// MARK: - 帧回调

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let handler = onFrame else { return }

        let isFront = currentInput?.device.position == .front
        let orientation: CGImagePropertyOrientation = isFront ? .leftMirrored : .right
        handler(pixelBuffer, orientation)
    }
}
