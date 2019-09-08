# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 下午1:14
# @Author  : Lart Pang
# @FileName: DrawFeatGrad.py
# @Project : Paper_Code
# @GitHub  : https://github.com/lartpang

from PIL import Image
from matplotlib import pyplot
from torch import nn
from torchvision.transforms import transforms

from Networks.WeakNet import TeacherNetV2
from Utils import joint_transforms

total_feat_out = []
total_feat_in = []
total_grad_in = []
total_grad_out = []


def hook_fn_forward(module, input, output):
    total_feat_in.append(input[0])
    total_feat_out.append(output)


def hook_fn_backward(module, grad_input, grad_output):
    # print('grad_output', grad_output[0].size(), len(grad_output))
    # print('grad_input', grad_input[0].size(), len(grad_input))
    # total_grad_in.append(grad_input[0])
    total_grad_out.append(grad_output[0])


net = TeacherNetV2()
modules = net.named_children()
for name, module in modules:
    module.register_forward_hook(hook_fn_forward)
    module.register_backward_hook(hook_fn_backward)
    print(name)

cal_loss = nn.BCELoss(reduction='mean')

x = Image.open('/home/lart/Datasets/RGBSaliency/DUTS/Test/Image/ILSVRC2012_test_00000003.jpg'
               '').convert('RGB')
y = Image.open('/home/lart/Datasets/RGBSaliency/DUTS/Test/Mask/ILSVRC2012_test_00000003.png'
               '').convert('L')
train_joint_transform = joint_transforms.Compose([
    joint_transforms.JointResize(320),  # R3Crop
])
train_img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 处理的是Tensor
])
train_target_transform = transforms.ToTensor()

x, y = train_joint_transform(x, y)
x = train_img_transform(x)
y = train_target_transform(y)
x = x.unsqueeze(0)
y = y.unsqueeze(0)
o = net(x)
loss = cal_loss(o, y)
loss.backward()

print('==========Saved inputs and outputs==========')
for idx in range(len(total_feat_in)):
    print('input: ', total_feat_in[idx].size())
    print('output: ', total_feat_out[idx].size())
    print('output: ', total_grad_out[idx].size())

for idx, feat_in in enumerate(total_feat_in):
    feat_in = feat_in.mean(1).detach().squeeze().numpy()
    feat_in = (feat_in - feat_in.min()) / (feat_in.max() - feat_in.min())
    pyplot.imshow(feat_in)
    pyplot.axis('off')

    # 去除坐标轴
    pyplot.savefig(f"./out_img/feat_{idx}.png", bbox_inches='tight', dpi=100, pad_inches=0)
    pyplot.show()

    print(f"{idx} is OK")

for idx, grad_out in enumerate(total_grad_out):
    grad_out = grad_out.mean(1).detach().squeeze().numpy()
    grad_out = (grad_out - grad_out.min()) / (grad_out.max() - grad_out.min())
    pyplot.imshow(grad_out)
    pyplot.axis('off')

    # 去除坐标轴
    pyplot.savefig(f"./out_img/grad_{idx}.png", bbox_inches='tight', dpi=100, pad_inches=0)
    pyplot.show()

    print(f"{idx} is OK")
