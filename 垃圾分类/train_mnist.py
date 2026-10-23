# train_mnist.py
import os
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore import nn, Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from models.lenet5 import LeNet5
from download import download

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def prepare_mnist_data():
    url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
    print("正在检查/下载 MNIST 数据集...")
    download(url, "./", kind="zip", replace=True)
    print("数据集准备完毕！")

def create_mnist_dataset(batch_size=64, usage='train'):
    data_path = os.path.join("./MNIST_Data", usage)
    dataset = ds.MnistDataset(dataset_dir=data_path, shuffle=(usage == 'train'))
    
    image_transforms = [
        vision.Resize((32, 32)),  # 确保统一缩放到 32x32 匹配 LeNet5 输入
        vision.Rescale(1.0/255.0, 0.0),
        vision.Normalize(mean=[0.1307], std=[0.3081]),
        vision.HWC2CHW()
    ]
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(C.TypeCast(ms.int32), 'label')
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

if __name__ == "__main__":
    prepare_mnist_data()
    
    print("正在加载数据集流...")
    train_dataset = create_mnist_dataset(batch_size=64, usage='train')
    test_dataset = create_mnist_dataset(batch_size=64, usage='test')
    
    network = LeNet5(num_class=10)
    
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    
    model = Model(network, net_loss, net_opt, metrics={"accuracy"})
    
    os.makedirs("./checkpoint", exist_ok=True)
    
    print("\n开始从头训练基础模型 LeNet5...")
    model.train(20, train_dataset, callbacks=[LossMonitor(per_print_times=100), TimeMonitor()])
    
    print("\n正在评估基础模型在 MNIST 测试集上的表现...")
    acc = model.eval(test_dataset)
    print(f"-> 基础模型测试集准确率: {acc['accuracy']*100:.2f}%")
    
    target_path = "./checkpoint/lenet5_mnist_final.ckpt"
    ms.save_checkpoint(network, target_path)
    print(f"\n预训练模型已成功保存至: {target_path}")