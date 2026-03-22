from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import onnx
import onnxruntime as ort
import torch


@dataclass
class ModelBundle:
    model_type: str
    model_path: Path
    torch_model: torch.nn.Module
    input_hw: Tuple[int, int]
    device: torch.device


class ONNXWrapper(torch.nn.Module):
    def __init__(self, onnx_path: Path, device: str = "cpu"):
        super().__init__()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: x_np},
        )[0]
        return torch.from_numpy(outputs).to(x.device)


def discover_model_paths(models_root: Path) -> Dict[str, Path]:
    return {p.stem: p for p in sorted(models_root.glob("*.onnx"))}


def infer_onnx_input_hw(onnx_path: Path) -> Tuple[int, int]:
    model = onnx.load(str(onnx_path))
    input0 = model.graph.input[0]
    dims = input0.type.tensor_type.shape.dim
    shape = [d.dim_value for d in dims]

    if len(shape) != 4:
        raise ValueError(f"Expected 4D input for {onnx_path}, got {shape}")

    h = int(shape[2]) if shape[2] > 0 else 224
    w = int(shape[3]) if shape[3] > 0 else 224
    return h, w


def load_model_bundle(model_type: str, model_path: Path, device: str = "cpu") -> ModelBundle:
    input_hw = infer_onnx_input_hw(model_path)
    model = ONNXWrapper(model_path, device=device).to(device)
    model.eval()

    return ModelBundle(
        model_type=model_type,
        model_path=model_path,
        torch_model=model,
        input_hw=input_hw,
        device=torch.device(device),
    )