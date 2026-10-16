import SwiftUI

/// 叠加层：在相机画面上画方框、标签和人体骨架。
///
/// 全部用 `Canvas` 绘制，不创建子视图，帧率高、开销小。
struct OverlayView: View {

    let items: [DetectedItem]
    let imageSize: CGSize
    /// 物体检测覆盖的区域（Vision 归一化坐标），nil 表示不画提示。
    let objectRegion: CGRect?

    /// 人物用青色，普通物体用琥珀色。
    private let personColor = Color(red: 0.15, green: 0.90, blue: 0.95)
    private let objectColor = Color(red: 1.00, green: 0.78, blue: 0.16)

    var body: some View {
        Canvas { context, size in
            guard imageSize.width > 0, imageSize.height > 0 else { return }

            if let region = objectRegion {
                drawRegionHint(region, in: &context, size: size)
            }

            for item in items {
                let rect = CoordinateMapper.viewRect(from: item.box,
                                                     imageSize: imageSize,
                                                     viewSize: size)
                guard rect.width > 2, rect.height > 2 else { continue }

                let color = item.isPerson ? personColor : objectColor

                if item.isPerson {
                    drawSkeleton(item, in: &context, size: size)
                }

                drawBox(rect, color: color, in: &context)
                drawLabel(item, above: rect, color: color, in: &context, viewSize: size)
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - 取景提示

    /// 用四个 L 形角标标出物体检测的有效范围。
    ///
    /// 物体检测走中心裁剪，上下边缘不参与推理；画出角标是为了让用户
    /// 知道把目标放在哪最准，而不是误以为「框不出来是坏了」。
    /// 画得很淡，检测到物体时也不会喧宾夺主。
    private func drawRegionHint(_ region: CGRect,
                                in context: inout GraphicsContext,
                                size: CGSize) {

        let rect = CoordinateMapper.viewRect(from: region,
                                             imageSize: imageSize,
                                             viewSize: size)
        guard rect.width > 40, rect.height > 40 else { return }

        let corner = min(24, min(rect.width, rect.height) * 0.18)

        var path = Path()
        // 左上
        path.move(to: CGPoint(x: rect.minX, y: rect.minY + corner))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.minX + corner, y: rect.minY))
        // 右上
        path.move(to: CGPoint(x: rect.maxX - corner, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY + corner))
        // 右下
        path.move(to: CGPoint(x: rect.maxX, y: rect.maxY - corner))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.maxX - corner, y: rect.maxY))
        // 左下
        path.move(to: CGPoint(x: rect.minX + corner, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY - corner))

        context.stroke(path,
                       with: .color(.white.opacity(0.32)),
                       style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
    }

    // MARK: - 方框

    private func drawBox(_ rect: CGRect, color: Color, in context: inout GraphicsContext) {
        let path = Path(roundedRect: rect, cornerRadius: 8)
        context.stroke(path, with: .color(.black.opacity(0.45)), lineWidth: 5)
        context.stroke(path, with: .color(color), lineWidth: 2.5)
    }

    // MARK: - 标签

    private func drawLabel(_ item: DetectedItem,
                           above rect: CGRect,
                           color: Color,
                           in context: inout GraphicsContext,
                           viewSize: CGSize) {

        let percent = Int((item.confidence * 100).rounded())
        let text = Text("\(item.label) \(percent)%")
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(.black)

        let resolved = context.resolve(text)
        let textSize = resolved.measure(in: CGSize(width: 320, height: 60))

        let paddingX: CGFloat = 6
        let paddingY: CGFloat = 4
        let chipWidth = textSize.width + paddingX * 2
        let chipHeight = textSize.height + paddingY * 2

        var originY = rect.minY - chipHeight - 4
        if originY < 0 {
            originY = rect.minY + 4
        }
        var originX = rect.minX
        if originX + chipWidth > viewSize.width {
            originX = max(0, viewSize.width - chipWidth)
        }

        let chip = CGRect(x: originX, y: originY, width: chipWidth, height: chipHeight)
        context.fill(Path(roundedRect: chip, cornerRadius: 5), with: .color(color))
        context.draw(resolved, at: CGPoint(x: chip.midX, y: chip.midY), anchor: .center)
    }

    // MARK: - 骨架

    private func drawSkeleton(_ item: DetectedItem,
                              in context: inout GraphicsContext,
                              size: CGSize) {

        var points: [Joint: CGPoint] = [:]
        for (joint, location) in item.joints {
            points[joint] = CoordinateMapper.viewPoint(from: location,
                                                        imageSize: imageSize,
                                                        viewSize: size)
        }

        // 骨骼连线
        let boneColor = Color(red: 1.0, green: 0.42, blue: 0.21)
        for (start, end) in skeletonBones {
            guard let from = points[start], let to = points[end] else { continue }
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path,
                           with: .color(boneColor),
                           style: StrokeStyle(lineWidth: 3, lineCap: .round))
        }

        // 关键点圆点
        for (_, point) in points {
            let dot = Path(ellipseIn: CGRect(x: point.x - 3.5,
                                             y: point.y - 3.5,
                                             width: 7,
                                             height: 7))
            context.fill(dot, with: .color(.white))
            context.stroke(dot, with: .color(boneColor), lineWidth: 1.5)
        }
    }
}
