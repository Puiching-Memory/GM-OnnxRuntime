import torch
import torch.nn as nn

# gamemaker GMlayer only supports double models

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(1, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 1, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      tensor_x = self.fc0(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc1(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc2(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      output = self.fc3(tensor_x)
      return output

model = MLPModel().double()
tensor_x = torch.tensor([0.5], dtype=torch.double)
onnx_program = torch.onnx.export(model, (tensor_x,), dynamo=True)
onnx_program.optimize()
onnx_program.save("mlp.onnx")

import onnxruntime
session = onnxruntime.InferenceSession("mlp.onnx")
print(session.run(None, {"tensor_x": tensor_x.numpy()}))
