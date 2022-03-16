import torch
import torchvision
import torch.nn as nn

import numpy as np
import onnx
from onnx import numpy_helper

def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
    # Reference : https://gaussian37.github.io/python-basic-numpy-snippets/
    flag = False
    try : 
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        print(layer_name + ": no difference.")
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag

# load onnx model
onnx_path = "ckpt/resnet18.onnx"
onnx_model = onnx.load(onnx_path)

# save onnx layer infos
onnx_layers = dict()
for layer in onnx_model.graph.initializer:
    onnx_layers[layer.name] = numpy_helper.to_array(layer)

# load torch model
torch_model = torchvision.models.resnet18()
torch_model.fc = nn.Linear(torch_model.fc.in_features, 10)
torch_path = "ckpt/resnet18.pth"
torch_model.load_state_dict(torch.load(torch_path))
torch_model.eval()

# save torch layer info
torch_layers = {}
for layer_name, layer_value in torch_model.named_modules():
    torch_layers[layer_name] = layer_value   


onnx_layers_set = set(onnx_layers.keys())

torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

for layer_name in filtered_onnx_layers:
    onnx_layer_name = layer_name
    torch_layer_name = layer_name.replace(".weight", "")
    onnx_weight = onnx_layers[onnx_layer_name]
    torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
    compare_two_array(onnx_weight, torch_weight, onnx_layer_name)