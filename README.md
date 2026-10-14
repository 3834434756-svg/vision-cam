# 视觉相机 VisionCam

一个 iOS 实时视觉 App：**摄像头画面里，物体自动画框标注名称，人物额外画出骨架**。
全部推理在本机完成，不联网、不上传。

<img src="docs/icon.png" width="120" alt="App 图标">

---

## 功能

| | 说明 |
|---|---|
| **物体识别** | YOLOv8n（CoreML，COCO 80 类）实时检测，画出方框 + 中文名称 + 置信度 |
| **人体姿态** | Vision `VNDetectHumanBodyPoseRequest`，19 个关键点 + 骨骼连线，附带人物外框 |
| **实时叠加** | 相机画面 + 检测结果叠加显示，底部可开关各项检测、调置信度门槛、切换前后摄像头 |
| **性能监控** | 顶部实时显示 FPS 与单帧推理耗时 |

界面配色约定：**青色框 = 人物**（带骨架），**黄色框 = 其他物体**。

## 技术方案

### 物体检测

用 YOLOv8n 的 CoreML 版本（Ultralytics 导出，**NMS 已内置在模型里**）：

```
输入  image       640×640 RGB
输出  confidence  [N, 80]    每行是一个框在 80 个类别上的置信度
      coordinates [N, 4]     每行 [cx, cy, w, h]，已归一化到 0~1
```

坐标格式依据 Apple CoreML 官方 proto（`NonMaximumSuppression.proto`）：
x/y 是**盒子中心点**，w/h 是宽高，且归一化到图像尺寸。

> **踩过的坑**：原始导出的模型除了 `image` 还带 `iouThreshold` /
> `confidenceThreshold` 两个必填标量输入，`VNCoreMLRequest` 无法提供，
> 会导致调用失败。本仓库内置的 `YOLOv8n.mlpackage` 已用 coremltools 打补丁
> 移除这两个输入（清空 `*InputFeatureName`），让 NMS 阶段回退使用内置参数
> （IoU 0.45 / 置信度 0.25），从而只剩单个 `image` 输入，Vision 可以直接跑。
> 打补丁的脚本见 `tools/patch_model.py`。

### 姿态检测

`VNDetectHumanBodyPoseRequest` 输出 19 个关键点（归一化坐标，原点左下），
直接连线画骨架；再由关键点外接框推导人物方框。

开启姿态检测时，物体检测会**跳过 `person` 类**，避免同一个人被画两个框。

### 坐标换算

Vision 坐标是「原点左下、归一化」的，预览层用 `.resizeAspectFill`。
`CoordinateMapper` 用同一套「等比放大 + 居中」把两边对齐，
所以框和骨架能严丝合缝地贴住画面里的物体。

App 锁定竖屏，因此方向映射是固定的：后置 `.right`、前置 `.leftMirrored`，
与预览层的默认显示保持一致。

## 目录结构

```
VisionCam/
├── project.yml                     # XcodeGen 工程描述（没有 .xcodeproj）
├── Support/                        # XcodeGen 生成的 Info.plist 落在这里
├── .github/workflows/
│   └── build-unsigned-ipa.yml      # 云端打未签名 IPA
├── tools/
│   └── patch_model.py              # 给 CoreML 模型打补丁（见上文）
└── VisionCam/
    ├── VisionCamApp.swift
    ├── ContentView.swift
    ├── Assets.xcassets/
    ├── Camera/
    │   ├── CameraManager.swift     # 采集、权限、前后摄切换、逐帧回调
    │   └── CameraPreview.swift     # AVCaptureVideoPreviewLayer 的 SwiftUI 封装
    ├── Detection/
    │   ├── DetectionTypes.swift    # DetectedItem / Joint / 骨骼连接表
    │   ├── COCOLabels.swift        # COCO 80 类中英文名
    │   ├── ObjectDetector.swift    # YOLOv8n + Vision，解析 NMS 输出
    │   ├── PoseDetector.swift      # 人体姿态
    │   └── DetectionEngine.swift   # 调度 + 背压 + 性能统计
    ├── UI/
    │   ├── CoordinateMapper.swift  # Vision 归一化坐标 → 屏幕坐标
    │   ├── OverlayView.swift       # Canvas 画框 / 标签 / 骨架
    │   └── ControlPanel.swift      # 底部控制面板
    └── Models/
        └── YOLOv8n.mlpackage       # 已打补丁的 CoreML 模型（6.2 MB）
```

## 构建

工程没有 `.xcodeproj`，用 [XcodeGen](https://github.com/yonaskolb/XcodeGen) 从
`project.yml` 现场生成：

```bash
brew install xcodegen
xcodegen generate
open VisionCam.xcodeproj
```

## 云端打包 IPA（不需要 Mac）

仓库自带 GitHub Actions workflow，推到 `main` 后自动构建：

```bash
gh run list --limit 3
gh run watch <run-id>
```

产物会出现在 **Releases** 页面（免登录下载）和当次运行的 Artifacts 区。

> ⚠️ 产出的是**未签名 IPA，不能直接安装**。需要用 Sideloadly / AltStore /
> SideStore 配合免费 Apple ID 重签，装完 **7 天**有效期，到期重签。

## 已知限制

- **未签名 IPA 有 7 天有效期**（免费 Apple ID 签名机制决定的，不是打包问题）。
- 物体检测精度受 YOLOv8n 这个最小模型限制，小目标 / 遮挡场景会漏检。
- 模型内部门槛固定为 0.25，界面上的置信度滑杆只能在此基础上**往上**过滤。
- App 锁定竖屏（横屏下坐标换算需要额外的方向处理）。

## 许可

仅供自用与学习。COCO 类别名来自公开数据集；YOLOv8n 权重版权归 Ultralytics
（AGPL-3.0），商用请自行确认授权。
