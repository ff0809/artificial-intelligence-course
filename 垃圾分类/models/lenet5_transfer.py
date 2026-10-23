# models/lenet5_transfer.py
import os
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal, Zero
from models.lenet5 import LeNet5

class LoRADense(nn.Cell):
    """密集连接全连接层 (nn.Dense) 的参数化高效低秩旁路适配器"""
    def __init__(self, original_dense, r=16, alpha=32):
        super(LoRADense, self).__init__()
        self.original = original_dense
        self.original.weight.requires_grad = False  
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        in_features = original_dense.in_channels
        out_features = original_dense.out_channels
        
        self.lora_A = Parameter(initializer(Normal(0.02), [r, in_features], ms.float32), name="lora_A", requires_grad=True)
        self.lora_B = Parameter(initializer(Zero(), [out_features, r], ms.float32), name="lora_B", requires_grad=True)
        self.scaling = alpha / r

    def construct(self, x):
        original_out = self.original(x)
        lora_out = ops.matmul(x, self.lora_A.T)
        lora_out = ops.matmul(lora_out, self.lora_B.T)
        return original_out + lora_out * self.scaling


class LoRAConv2d(nn.Cell):
    """专门针对视觉卷积层 (nn.Conv2d) 研发的低秩分解外挂适配器"""
    def __init__(self, original_conv, r=16, alpha=32):
        super(LoRAConv2d, self).__init__()
        self.original = original_conv
        self.original.weight.requires_grad = False  
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size[0]
        stride = original_conv.stride[0]
        pad_mode = original_conv.pad_mode
        
        self.lora_A = nn.Conv2d(in_channels, r, kernel_size, stride=stride, 
                                pad_mode=pad_mode, has_bias=False, weight_init=Normal(0.02))
        self.lora_B = nn.Conv2d(r, out_channels, 1, stride=1, 
                                pad_mode='valid', has_bias=False, weight_init=Zero())
        self.scaling = alpha / r

    def construct(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(x))
        return original_out + lora_out * self.scaling


class LeNet5Transfer(nn.Cell):
    """迁移重构型的通用 LeNet5 微调网络骨架"""
    def __init__(self, num_classes=26, checkpoint_path="./checkpoint/lenet5_mnist_final.ckpt", lora_mode="none"):
        super(LeNet5Transfer, self).__init__()
        base_net = LeNet5(num_class=10)
        
        actual_ckpt = checkpoint_path
        if not os.path.exists(actual_ckpt) and os.path.exists("./checkpoints/lenet5_mnist.ckpt"):
            actual_ckpt = "./checkpoints/lenet5_mnist.ckpt"
            
        if os.path.exists(actual_ckpt):
            param_dict = ms.load_checkpoint(actual_ckpt)
            ms.load_param_into_net(base_net, param_dict, strict_load=False)
            print(f"成功加载 MNIST 预训练权重: {actual_ckpt}")
        else:
            print(f"未找到预训练文件，采用基础随机初始化")
            
        conv1_layer = base_net.conv1
        conv2_layer = base_net.conv2
        fc1_layer = base_net.fc1
        fc2_layer = base_net.fc2
        
        if lora_mode == "dense_only":
            print("弹性LoRA：正在仅为密集连接层 (fc1+fc2) 挂载低秩旁路...")
            fc1_layer = LoRADense(fc1_layer, r=16, alpha=32)
            fc2_layer = LoRADense(fc2_layer, r=16, alpha=32)
            
        elif lora_mode == "all_layers":
            print("全线LoRA高级改进：正在将低秩矩阵全线覆盖至 [视觉卷积层+全连接层]...")
            fc1_layer = LoRADense(fc1_layer, r=16, alpha=32)
            fc2_layer = LoRADense(fc2_layer, r=16, alpha=32)
            conv1_layer = LoRAConv2d(conv1_layer, r=16, alpha=32)  
            conv2_layer = LoRAConv2d(conv2_layer, r=16, alpha=32)  
            
        self.backbone = nn.CellList([
            conv1_layer, base_net.pool1,
            conv2_layer, base_net.pool2,
            base_net.flatten, fc1_layer, fc2_layer
        ])
        self.relu = nn.ReLU()
        self.fc3_garbage = nn.Dense(84, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.backbone[1](self.relu(self.backbone[0](x))) 
        x = self.backbone[3](self.relu(self.backbone[2](x))) 
        x = self.backbone[4](x)                              
        x = self.relu(self.backbone[5](x))                   
        x = self.relu(self.backbone[6](x))                   
        return self.fc3_garbage(x)


class LeNet5StructureImproved(nn.Cell):
    """【模型改进方案】全量微调下的内置 BatchNorm+残差对齐+LeakyReLU 改进网络"""
    def __init__(self, num_classes=26, checkpoint_path="./checkpoint/lenet5_mnist_final.ckpt"):
        super(LeNet5StructureImproved, self).__init__()
        base_net = LeNet5(num_class=10)
        
        actual_ckpt = checkpoint_path
        if not os.path.exists(actual_ckpt) and os.path.exists("./checkpoints/lenet5_mnist.ckpt"):
            actual_ckpt = "./checkpoints/lenet5_mnist.ckpt"
            
        if os.path.exists(actual_ckpt):
            param_dict = ms.load_checkpoint(actual_ckpt)
            ms.load_param_into_net(base_net, param_dict, strict_load=False)
            print(f"[INFO] 结构改进模型成功加载骨干参数。")

        self.conv1 = base_net.conv1
        self.pool1 = base_net.pool1
        self.conv2 = base_net.conv2
        self.pool2 = base_net.pool2
        self.flatten = base_net.flatten
        self.fc1 = base_net.fc1
        self.fc2 = base_net.fc2
        self.fc3_garbage = nn.Dense(84, num_classes)
        
        self.bn1 = nn.BatchNorm2d(6)   
        self.bn2 = nn.BatchNorm2d(16)  
        self.leaky_relu = nn.LeakyReLU(alpha=0.1) 
        
        self.shortcut_align = nn.Conv2d(6, 16, 5, pad_mode='valid', has_bias=False)
        self.bn_short = nn.BatchNorm2d(16)

    def construct(self, x):
        x1 = self.pool1(self.leaky_relu(self.bn1(self.conv1(x))))
        identity = self.bn_short(self.shortcut_align(x1))
        x2 = self.bn2(self.conv2(x1))
        x2 = self.pool2(self.leaky_relu(x2 + identity))
        x = self.flatten(x2)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        return self.fc3_garbage(x)


class LeNet5StructureAdvanced(nn.Cell):
    """【高级模型改进方案】MaxPool+BN+残差+空间维度扩容 最终改进版本 LeNet5"""
    def __init__(self, num_classes=26):
        super(LeNet5StructureAdvanced, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid', has_bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid', has_bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.shortcut_align = nn.Conv2d(6, 16, 5, pad_mode='valid', has_bias=False)
        self.bn_short = nn.BatchNorm2d(16)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Dense(400, 256, weight_init=Normal(0.02), has_bias=False)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        self.fc2_expanded = nn.Dense(256, 128, weight_init=Normal(0.02), has_bias=False) 
        self.bn_fc2 = nn.BatchNorm1d(128)
        
        self.fc3_garbage = nn.Dense(128, num_classes, weight_init=Normal(0.02))
        self.leaky_relu = nn.LeakyReLU(alpha=0.1)
        
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        x1 = self.pool1(self.leaky_relu(self.bn1(self.conv1(x))))
        
        identity = self.bn_short(self.shortcut_align(x1))
        x2 = self.bn2(self.conv2(x1))
        
        x2 = self.pool2(self.leaky_relu(x2 + identity))
        
        x = self.flatten(x2)
        x = self.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.leaky_relu(self.bn_fc2(self.fc2_expanded(x)))
        x = self.dropout(x)
        return self.fc3_garbage(x)