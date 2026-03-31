# export_deit_tiny_to_onnx.py
import torch
import timm

model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "DeiT_tiny.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
)
print("saved: DeiT_tiny.onnx")
