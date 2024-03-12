"""
本代码使用torchvision中的预训练模型进行模型转换例程
参考代码: https://github.com/ultralytics/yolov5/blob/master/export.py
转换为engine文件的思路如下: pth/pt/pkl -> onnx -> engine
可以在x86下先将模型文件转换为onnx, 再把onnx传到板子上转engine

请注意, 由于x86和arm上cuda device架构不同, 不允许将x86下转换
的engine文件部署到arm上运行

在分类模型中输入图像为3x640x640
在回归模型中输入图像为3x224x224
"""

import torch
import onnx
import tensorrt as trt
from script.Model import ResNet, ResNet18, VGG11
from torchvision.models import resnet18, vgg11

# model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

device = torch.device(0)
# load model
model_path = "./resnet18_epoch_95.pkl"

base_model = model_factory['resnet18']
regressor = regressor_factory['resnet18'](model=base_model)

checkpoint = torch.load(model_path)

regressor.load_state_dict(checkpoint['model_state_dict'])
regressor.to(device)
regressor.eval()

# to onnx
onnx_file = model_path.replace("pkl", "onnx")
output_names = ["orient", "conf", "dim"]
img = torch.zeros(1, 3, 224, 224).to(device)
torch.onnx.export(regressor, img, onnx_file, verbose=False, opset_version=12, do_constant_folding=True, input_names=['imgaes'], output_names=output_names)

# checks
model_onnx = onnx.load(onnx_file)
onnx.checker.check_model(model_onnx)
# d = {'stride': int(max(regressor.stride)), 'names': regressor.names}
# for k, v in d.items():
#     meta = model_onnx.metadata_props.add()
#     meta.key, meta.value = k, str(v)
onnx.save(model_onnx, onnx_file)

# to engine
engine_file = model_path.replace("pkl", "engine")
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
config = builder.create_builder_config()
workspace = 4
config.max_workspace_size = workspace * 1 << 28

flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(onnx_file):
    raise RuntimeError(f'failed to load ONNX file: {onnx_file}')

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]

for inp in inputs:
    print(f'input {inp.name} with shape {inp.shape} {inp.dtype}')
for out in outputs:
    print(f'output {out.name} with shape {out.shape} {out.dtype}')

half = False
if builder.platform_has_fast_fp16 and half:
    config.set_flag(trt.BuilderFlag.FP16)


with builder.build_engine(network, config) as engine, open(engine_file, 'wb') as t:
    t.write(engine.serialize())
