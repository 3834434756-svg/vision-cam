"""给 Ultralytics 导出的 YOLO CoreML pipeline 打补丁。

Ultralytics 的 CoreML 导出会额外暴露 `iouThreshold` / `confidenceThreshold`
两个**必填**标量输入，而 `VNCoreMLRequest` 无法提供这类输入，直接调用会失败。
本脚本把这两个输入从顶层描述里删掉，并清空 NMS 阶段对它们的引用，
让 NMS 回退使用内置参数，从而模型只剩单个 `image` 输入。

用法:
    python3 patch_model.py <src.mlpackage> <dst.mlpackage>
"""

import sys

import coremltools as ct

DROPPED_INPUTS = ("iouThreshold", "confidenceThreshold")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    src, dst = sys.argv[1], sys.argv[2]

    model = ct.models.MLModel(src, skip_model_load=True)
    spec = model.get_spec()

    print("原始输入:", [i.name for i in spec.description.input])
    print("原始输出:", [o.name for o in spec.description.output])

    nms = spec.pipeline.models[1].nonMaximumSuppression
    nms.ClearField("iouThresholdInputFeatureName")
    nms.ClearField("confidenceThresholdInputFeatureName")
    print("NMS 内置阈值 iou/conf:", nms.iouThreshold, "/", nms.confidenceThreshold)

    # 按名字删除，避免依赖输入顺序
    indices = [i for i, feature in enumerate(spec.description.input)
               if feature.name in DROPPED_INPUTS]
    for index in reversed(indices):
        del spec.description.input[index]

    print("补丁后输入:", [i.name for i in spec.description.input])

    patched = ct.models.MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
    patched.save(dst)

    # 回读校验
    check = ct.models.MLModel(dst, skip_model_load=True).get_spec()
    check_nms = check.pipeline.models[1].nonMaximumSuppression

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
    print("保留的内置阈值:", check_nms.iouThreshold, check_nms.confidenceThreshold)
    print("类别数:", len(check_nms.stringClassLabels.vector))

    single_input = len(check.description.input) == 1
    print("结论:", "单输入 image，Vision 可直接调用 ✅" if single_input
          else "仍有多个输入，需要检查 ❌")
    return 0 if single_input else 1


if __name__ == "__main__":
    raise SystemExit(main())
