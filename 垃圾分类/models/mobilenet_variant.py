import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

class ConvBNReLU6(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU6, self).__init__()
        padding = (kernel_size - 1) // 2
        # 使用组卷积 (groups) 来兼容普通卷积与深度可分离卷积 (Depthwise)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, group=groups, has_bias=False,
                              weight_init=Normal(0.02))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()

    def construct(self, x):
        return self.relu(self.bn(self.conv(x)))


class InvertedResidual(nn.Cell):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # 当步长为 1 且输入输出通道相同时，开启残差连接 (Shortcut)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        # 1. 扩张阶段 (1x1 点对点卷积)：将低维特征映射到高维空间
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
        
        # 2. 深度卷积阶段 (3x3 逐通道卷积)：在特征高维空间极其廉价地提取空间特征
        layers.append(ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        
        # 3. 投影阶段 (1x1 降维线性卷积)：注意为了防止破坏高维特征，末尾绝对不加 ReLU6 激活函数
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, stride=1, pad_mode='pad', padding=0, has_bias=False, weight_init=Normal(0.02)))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Garbage(nn.Cell):
    def __init__(self, num_classes=26, width_mult=1.0):
        super(MobileNetV2Garbage, self).__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2], # 尺寸减半
            [6, 32, 3, 2], # 尺寸减半
            [6, 64, 4, 2], # 尺寸减半
            [6, 96, 3, 1],
            [6, 160, 3, 2], # 尺寸减半
            [6, 320, 1, 1],
        ]

        # 1. 网络的最初始特征提取层 (Stem Stage)
        features = [ConvBNReLU6(3, input_channel, kernel_size=3, stride=2)]

        # 2. 按照官方配置表组装繁复的倒残差骨干特征提取网 (Backbone)
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 3. 末尾的 1x1 卷积升维层
        features.append(ConvBNReLU6(input_channel, last_channel, kernel_size=1))
        self.features = nn.SequentialCell(features)

        # 4. 分类器预测头：将 1280 维全局特征映射为 26 个垃圾类别提示
        self.classifier = nn.Dense(last_channel, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        # 提取高阶多维图像空间特征
        x = self.features(x)
        # 执行全局空间平均池化 (通过在空间维度 2 和 3 上取均值实现降维)
        x = ops.mean(x, (2, 3))
        # 送入最终的全连接分类器
        x = self.classifier(x)
        return x