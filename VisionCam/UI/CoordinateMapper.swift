import CoreGraphics

/// Vision 归一化坐标 → 屏幕坐标的换算。
///
/// Vision 给的坐标：原点在**左下角**，x/y ∈ [0, 1]，对应已摆正的图像。
/// 预览层用的是 `.resizeAspectFill`：图像等比放大到填满视图，超出部分裁掉。
/// 两边都按同一套「等比放大 + 居中」算，框就不会错位。
enum CoordinateMapper {

    /// 把归一化矩形换算成视图坐标（原点左上）。
    static func viewRect(from rect: CGRect, imageSize: CGSize, viewSize: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0,
              viewSize.width > 0, viewSize.height > 0 else {
            return .zero
        }

        let scale = max(viewSize.width / imageSize.width,
                        viewSize.height / imageSize.height)
        let displayWidth = imageSize.width * scale
        let displayHeight = imageSize.height * scale
        let offsetX = (viewSize.width - displayWidth) / 2
        let offsetY = (viewSize.height - displayHeight) / 2

        return CGRect(
            x: offsetX + rect.minX * displayWidth,
            y: offsetY + (1 - rect.maxY) * displayHeight,   // y 轴翻转
            width: rect.width * displayWidth,
            height: rect.height * displayHeight
        )
    }

    /// 把归一化点换算成视图坐标（原点左上）。
    static func viewPoint(from point: CGPoint, imageSize: CGSize, viewSize: CGSize) -> CGPoint {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }

        let scale = max(viewSize.width / imageSize.width,
                        viewSize.height / imageSize.height)
        let displayWidth = imageSize.width * scale
        let displayHeight = imageSize.height * scale
        let offsetX = (viewSize.width - displayWidth) / 2
        let offsetY = (viewSize.height - displayHeight) / 2

        return CGPoint(
            x: offsetX + point.x * displayWidth,
            y: offsetY + (1 - point.y) * displayHeight
        )
    }
}
