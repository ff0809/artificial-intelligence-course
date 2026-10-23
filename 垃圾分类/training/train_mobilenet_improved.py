import os
import sys
import time
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mobilenet_variant import MobileNetV2Garbage
from garbage_dataset import create_garbage_dataset

ms.set_seed(42)
np.random.seed(42)

def run_mobilenet_experiment(epochs=20):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", deterministic="ON")
    real_data_dir = "./data/data_en"
    
    # 独立呼叫3通道 224x224 彩色大图输入
    train_loader, num_classes = create_garbage_dataset(real_data_dir, batch_size=32, training=True, model_type="mobilenet")
    test_loader, _ = create_garbage_dataset(real_data_dir, batch_size=32, training=False, model_type="mobilenet")
    
    net = MobileNetV2Garbage(num_classes=num_classes)
    
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=2e-4, weight_decay=1e-4)
    
    def forward_fn(imgs, lbls):
        return loss_fn(net(imgs), lbls)
    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    mem_start = psutil.Process().memory_info().rss / 1024 / 1024
    time_start = time.time()
    
    print(f"开始训练MobileNetV2，共规划 {epochs} 个Epoch...")
    for epoch in range(epochs):
        net.set_train(True)
        epoch_loss = []
        for step, (images, labels) in enumerate(train_loader.create_tuple_iterator()):
            loss, grads = grad_fn(images, labels)
            grads = ops.clip_by_global_norm(grads, clip_norm=1.0) # 全局梯度裁剪
            optimizer(grads)
            epoch_loss.append(loss.asnumpy())
            if step % 20 == 0:
                print(f"  -> Epoch [{epoch+1}/{epochs}] Step [{step}] 实时单步 Loss: {loss.asnumpy():.4f}")
        print(f" -> Epoch [{epoch+1}/{epochs}] 完成，本轮平均 Loss: {np.mean(epoch_loss):.4f}")
        
    time_elapsed = time.time() - time_start
    mem_end = psutil.Process().memory_info().rss / 1024 / 1024
    memory_increase = max(0.0, mem_end - mem_start)
    
    print("训练结束，正在进行结果度量...")
    net.set_train(False)
    correct, total = 0, 0
    for data in test_loader.create_dict_iterator():
        logits = net(data['image'])
        correct += (ops.argmax(logits, dim=1) == data['label']).asnumpy().sum()
        total += data['label'].shape[0]
    final_acc = correct / total
    
    print(f"MobileNetV2 测试集最终准确率: {final_acc*100:.2f}% | 耗时: {time_elapsed:.2f}s | 内存增量: {memory_increase:.2f}MB")
    return final_acc, time_elapsed, memory_increase

if __name__ == "__main__":
    run_mobilenet_experiment(epochs=20)