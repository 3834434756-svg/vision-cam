"""调整 CoreML 模型内置 NMS 的阈值（原地修改 mlpackage 里的 model.mlmodel）。

Ultralytics 导出的 YOLO pipeline 把 NMS 的置信度门槛写死在模型里（默认 0.25），
App 端的置信度滑杆只能在它之上做二次过滤 —— 低于 0.25 的目标永远出不来。
把内置门槛降到 0.10，滑杆才能真正往下调，接住杯子、书本、手机这类弱目标；
同时把 IoU 从 0.7 降到 0.45，减少同一个物体被画好几个框。

只改 model.mlmodel 这份 proto，不碰 weights/weight.bin，
所以不会影响权重、也不需要重新导出模型。

用法:
    python3 tools/tune_nms.py VisionCam/Models/YOLO11m.mlpackage --conf 0.10 --iou 0.45
"""

import argparse
import pathlib
import sys

from coremltools.proto import Model_pb2

SPEC_PATH = "Data/com.apple.CoreML/model.mlmodel"
DROPPED_INPUTS = ("iouThreshold", "confidenceThreshold")


def load_spec(mlmodel_path: pathlib.Path) -> Model_pb2.Model:
    spec = Model_pb2.Model()
    spec.ParseFromString(mlmodel_path.read_bytes())
    return spec


def find_nms(spec: Model_pb2.Model):
    if not spec.HasField("pipeline"):
        return None
    for stage in spec.pipeline.models:
        if stage.HasField("nonMaximumSuppression"):
            return stage.nonMaximumSuppression
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=pathlib.Path, help="YOLO*.mlpackage 路径")
    parser.add_argument("--conf", type=float, default=0.10, help="NMS 置信度门槛，默认 0.10")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 门槛，默认 0.45")
    args = parser.parse_args()

    mlmodel_path = args.package / SPEC_PATH
    if not mlmodel_path.exists():
        print(f"找不到 {mlmodel_path}", file=sys.stderr)
        return 2

    spec = load_spec(mlmodel_path)
    nms = find_nms(spec)
    if nms is None:
        print("这个模型里没有 NMS 阶段，不需要调整。", file=sys.stderr)
        return 2

    print("调整前: conf =", nms.confidenceThreshold, " iou =", nms.iouThreshold)
    nms.confidenceThreshold = args.conf
    nms.iouThreshold = args.iou
    # 运行时覆盖输入在 patch_model.py 里已经清掉，这里再兜一次底
    nms.ClearField("iouThresholdInputFeatureName")
    nms.ClearField("confidenceThresholdInputFeatureName")

    # 顺手把导出时写进 metadata 的说明也改掉，免得 Xcode 里看到的还是旧值
    metadata = spec.description.metadata.userDefined
    if "IoU threshold" in metadata:
        metadata["IoU threshold"] = f"{args.iou:.2f}"
    if "Confidence threshold" in metadata:
        metadata["Confidence threshold"] = f"{args.conf:.2f}"

    indices = [i for i, feature in enumerate(spec.description.input)
               if feature.name in DROPPED_INPUTS]
    for index in reversed(indices):
        del spec.description.input[index]

    mlmodel_path.write_bytes(spec.SerializeToString())

    # 回读校验
    check = load_spec(mlmodel_path)
    check_nms = find_nms(check)
    print("调整后: conf =", check_nms.confidenceThreshold, " iou =", check_nms.iouThreshold)
    print("输入:", [i.name for i in check.description.input])
    print("输出:", [o.name for o in check.description.output])
    print("结论: ✅" if check_nms.confidenceThreshold == args.conf else "结论: ❌")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
