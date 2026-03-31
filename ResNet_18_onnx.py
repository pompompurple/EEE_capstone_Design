# export_resnet18_to_onnx.py
import torch
import torchvision.models as models

model = models.resnet18(weights=None)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "ResNet_18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
)
print("saved: ResNet_18.onnx")
