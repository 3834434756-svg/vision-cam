"""给 YOLOv8n CoreML pipeline 打补丁：
移除 iouThreshold / confidenceThreshold 两个必填标量输入，
让 NMS 阶段回退使用其内置参数（0.45 / 0.25）。
这样模型只剩单个 image 输入，可被 Vision 的 VNCoreMLRequest 直接调用。
"""
import shutil
import coremltools as ct

SRC = "/workspace/VisionCam/VisionCam/Models/YOLOv8n.mlpackage"
DST = "/root/.codebuddy/artifact/2f8a7d30-9c21-4f0b-1d49-944032875000/YOLOv8n.patched.mlpackage"

m = ct.models.MLModel(SRC, skip_model_load=True)
spec = m.get_spec()

print("原始输入:", [i.name for i in spec.description.input])
print("原始输出:", [o.name for o in spec.description.output])

nms = spec.pipeline.models[1].nonMaximumSuppression
nms.ClearField("iouThresholdInputFeatureName")
nms.ClearField("confidenceThresholdInputFeatureName")
print("NMS iou 参数:", nms.iouThreshold, "| conf 参数:", nms.confidenceThreshold)

del spec.description.input[1]  # iouThreshold
del spec.description.input[1]  # confidenceThreshold
print("补丁后输入:", [i.name for i in spec.description.input])

new = ct.models.MLModel(spec, weights_dir=m.weights_dir, skip_model_load=True)
new.save(DST)

# 回读校验
chk = ct.models.MLModel(DST, skip_model_load=True)
cs = chk.get_spec()
print()
print("=== 回读校验 ===")
print("输入:", [(i.name, i.type.WhichOneof('Type')) for i in cs.description.input])
print("输出:", [(o.name, o.type.WhichOneof('Type')) for o in cs.description.output])
print("pipeline 阶段数:", len(cs.pipeline.models))
c2 = cs.pipeline.models[1].nonMaximumSuppression
print("iouThresholdInputFeatureName 已清空:", not c2.HasField("iouThresholdInputFeatureName"))
print("confidenceThresholdInputFeatureName 已清空:", not c2.HasField("confidenceThresholdInputFeatureName"))
print("保留的阈值参数:", c2.iouThreshold, c2.confidenceThreshold)
print("类别数:", len(c2.stringClassLabels.vector))
