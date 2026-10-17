"""给 Ultralytics 导出的 YOLO CoreML pipeline 打补丁。

做两件事：

1. Ultralytics 的 CoreML 导出会额外暴露 `iouThreshold` / `confidenceThreshold`
   两个**必填**标量输入，而 `VNCoreMLRequest` 无法提供这类输入，直接调用会失败。
   本脚本把这两个输入从顶层描述里删掉，并清空 NMS 阶段对它们的引用，
   让 NMS 回退使用内置参数，从而模型只剩单个 `image` 输入。
2. 把内置 NMS 的置信度门槛降到 0.10、IoU 降到 0.45。默认 0.25 的门槛会让
   杯子、书本这类弱目标在模型内部就被丢掉，App 端滑杆再低也没用。

用法:
    python3 patch_model.py <src.mlpackage> <dst.mlpackage> [--conf 0.10] [--iou 0.45]
"""

import argparse
import sys

import coremltools as ct

DROPPED_INPUTS = ("iouThreshold", "confidenceThreshold")


def find_nms(spec):
    if not spec.HasField("pipeline"):
        return None
    for stage in spec.pipeline.models:
        if stage.HasField("nonMaximumSuppression"):
            return stage.nonMaximumSuppression
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    model = ct.models.MLModel(args.src, skip_model_load=True)
    spec = model.get_spec()

    print("原始输入:", [i.name for i in spec.description.input])
    print("原始输出:", [o.name for o in spec.description.output])

    nms = find_nms(spec)
    if nms is None:
        print("找不到 NMS 阶段，模型结构与预期不符 ❌", file=sys.stderr)
        return 1

    nms.ClearField("iouThresholdInputFeatureName")
    nms.ClearField("confidenceThresholdInputFeatureName")
    nms.confidenceThreshold = args.conf
    nms.iouThreshold = args.iou
    print("NMS 阈值 iou/conf:", nms.iouThreshold, "/", nms.confidenceThreshold)

    metadata = spec.description.metadata.userDefined
    if "IoU threshold" in metadata:
        metadata["IoU threshold"] = f"{args.iou:.2f}"
    if "Confidence threshold" in metadata:
        metadata["Confidence threshold"] = f"{args.conf:.2f}"

    # 按名字删除，避免依赖输入顺序
    indices = [i for i, feature in enumerate(spec.description.input)
               if feature.name in DROPPED_INPUTS]
    for index in reversed(indices):
        del spec.description.input[index]

    print("补丁后输入:", [i.name for i in spec.description.input])

    patched = ct.models.MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
    patched.save(args.dst)

    # 回读校验
    check = ct.models.MLModel(args.dst, skip_model_load=True).get_spec()
    check_nms = find_nms(check)

    print()
    print("=== 回读校验 ===")
    print("输入:", [i.name for i in check.description.input])
    print("输出:", [o.name for o in check.description.output])
    print("pipeline 阶段数:", len(check.pipeline.models))
    # proto3 的 string 字段没有 presence，用空串判断是否已清空
    print("iouThresholdInputFeatureName 已清空:",
          check_nms.iouThresholdInputFeatureName == "")
    print("confidenceThresholdInputFeatureName 已清空:",
          check_nms.confidenceThresholdInputFeatureName == "")
    print("内置阈值 iou/conf:", check_nms.iouThreshold, "/", check_nms.confidenceThreshold)
    print("类别数:", len(check_nms.stringClassLabels.vector))

    single_input = len(check.description.input) == 1
    print("结论:", "单输入 image，Vision 可直接调用 ✅" if single_input
          else "仍有多个输入，需要检查 ❌")
    return 0 if single_input else 1


if __name__ == "__main__":
    raise SystemExit(main())
