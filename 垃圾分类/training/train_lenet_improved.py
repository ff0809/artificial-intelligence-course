import os
import sys
import time
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lenet5_transfer import LeNet5Transfer, LeNet5StructureAdvanced as FinalLeNetStructureImproved
from garbage_dataset import create_garbage_dataset

ms.set_seed(42)
np.random.seed(42)

def run_comparison_experiment(strategy="baseline", epochs=100):
    ms.set_seed(42)
    np.random.seed(42)
    
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", deterministic="ON")
    real_data_dir = "./data/data_en"
    
    train_loader, num_classes = create_garbage_dataset(real_data_dir, batch_size=32, training=True, model_type="lenet5")
    test_loader, _ = create_garbage_dataset(real_data_dir, batch_size=32, training=False, model_type="lenet5")
    
    if strategy == "baseline":
        print(f"\n运行传统LeNet5迁移网络（改进前） ========")
        net = LeNet5Transfer(num_classes=num_classes, lora_mode="none")
    else:
        print(f"\n运行内置 MaxPool+BatchNorm+残差对齐+空间扩容LeNet5（改进后） ========")
        net = FinalLeNetStructureImproved(num_classes=num_classes)
        
    for param in net.get_parameters():
        param.requires_grad = True
        
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-4, weight_decay=5e-4)
    
    def forward_fn(imgs, lbls):
        return loss_fn(net(imgs), lbls)
    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    mem_start = psutil.Process().memory_info().rss / 1024 / 1024
    time_start = time.time()
    
    print(f"开始训练，共规划 {epochs} 个Epoch...")
    for epoch in range(epochs):
        net.set_train(True)
        epoch_loss = []
        for images, labels in train_loader.create_tuple_iterator():
            loss, grads = grad_fn(images, labels)
            grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
            optimizer(grads)
            epoch_loss.append(loss.asnumpy())
        print(f" -> Epoch [{epoch+1}/{epochs}] 完成，平均训练 Loss: {np.mean(epoch_loss):.4f}")
            
    time_elapsed = time.time() - time_start
    # 计算物理内存占用
    mem_end = psutil.Process().memory_info().rss / 1024 / 1024
    memory_increase = max(0.0, mem_end - mem_start)
    
    net.set_train(False)
    correct, total = 0, 0
    for data in test_loader.create_dict_iterator():
        logits = net(data['image'])
        correct += (ops.argmax(logits, dim=1) == data['label']).asnumpy().sum()
        total += data['label'].shape[0]
    final_acc = correct / total
    
    print(f"策略: {strategy} | 测试集最终准确率: {final_acc*100:.2f}% | 耗时: {time_elapsed:.2f}s | 内存增量: {memory_increase:.2f}MB")
    return final_acc, time_elapsed, memory_increase

if __name__ == "__main__":
    acc_base, time_base, mem_base = run_comparison_experiment(strategy="baseline", epochs=100)
    print("\n" + "-"*50 + "\n")
    acc_imp, time_imp, mem_imp = run_comparison_experiment(strategy="structure_improved", epochs=100)
    
    print("\n" + "="*53)
    print("模型架构改进结果对比")
    print("="*53)
    print(f"{'网络拓扑架构策略 (超参数全对齐)':<38} | {'测试集最终准确率(%)':<15} | {'总耗时(s)':<10} | {'内存/显存增量(MB)':<15}")
    print("-" * 53)
    print(f"{'LeNet5 传统串行迁移架构 (改进前)':<38} | {acc_base*100:<15.2f} | {time_base:<10.2f} | {mem_base:<15.2f}")
    print(f"{'LeNet5 MaxPool+BN+残差对齐+维度扩容 (改进后)':<38} | {acc_imp*100:<15.2f} | {time_imp:<10.2f} | {mem_imp:<15.2f}")
    print("="*53)