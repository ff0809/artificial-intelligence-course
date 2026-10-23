import os
import sys
import time
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lenet5_transfer import LeNet5Transfer
from garbage_dataset import create_garbage_dataset

# 设置全局随机种子，确保实验可复现
ms.set_seed(42)
np.random.seed(42)

def run_lora_experiment(strategy="dense_only", epochs=100):
    ms.set_seed(42)
    np.random.seed(42)
    
    # 开启底层硬件执行确定性锁死，严防结果抖动
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", deterministic="ON")
    real_data_dir = "./data/data_en"
    
    train_loader, num_classes = create_garbage_dataset(real_data_dir, batch_size=32, training=True, model_type="lenet5")
    test_loader, _ = create_garbage_dataset(real_data_dir, batch_size=32, training=False, model_type="lenet5")
    
    net = LeNet5Transfer(num_classes=num_classes, lora_mode=strategy)
    
    # 梯度权限划分
    if strategy == "dense_only":
        for param in net.get_parameters():
            if "lora_" in param.name or "fc3_garbage" in param.name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif strategy == "all_layers":
        for param in net.get_parameters():
            if "lora_" in param.name or "fc3_garbage" in param.name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                    
    # 优化器配置 
    head_params = []
    lora_params = []
    for param in net.trainable_params():
        if "fc3_garbage" in param.name:
            head_params.append(param)
        else:
            lora_params.append(param)
            
    optimizer = nn.Adam([
        {"params": lora_params, "lr": 2e-4},
        {"params": head_params, "lr": 6e-4}
    ], weight_decay=1e-3)
    
    print(f"总参数数量: {len(list(net.get_parameters()))} | 本次参与训练更新的参数数量: {len(net.trainable_params())}")
    print(f"当前策略: {strategy}")
            
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
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
    
    mem_end = psutil.Process().memory_info().rss / 1024 / 1024
    memory_increase = max(0.0, mem_end - mem_start)
    
    net.set_train(False)
    correct, total = 0, 0
    for data in test_loader.create_dict_iterator():
        logits = net(data['image'])
        correct += (ops.argmax(logits, dim=1) == data['label']).asnumpy().sum()
        total += data['label'].shape[0]
    final_acc = correct / total
    
    print(f"策略: {strategy} | 测试集最终准确率: {final_acc*100:.2f}% | 耗时: {time_elapsed:.2f}s | 显存/内存增量: {memory_increase:.2f}MB")
    return final_acc, time_elapsed, memory_increase

if __name__ == "__main__":
    acc_dense, time_dense, mem_dense = run_lora_experiment(strategy="dense_only", epochs=100)
    print("\n" + "-"*50 + "\n")
    acc_all, time_all, mem_all = run_lora_experiment(strategy="all_layers", epochs=100)
    
    print("\n" + "="*65)
    print("LoRA微调结果对比")
    print("="*65)
    print(f"{'LoRA 微调覆盖层级策略 (统一基座 LeNet5)':<38} | {'测试集最终准确率(%)':<15} | {'总耗时(s)':<10} | {'内存/显存增量(MB)':<15}")
    print("-" * 65)
    print(f"{'传统 LoRA 微调 (仅 Dense 层挂载)':<38} | {acc_dense*100:<15.2f} | {time_dense:<10.2f} | {mem_dense:<15.2f}")
    print(f"{'全线层级覆盖 LoRA 改进微调 (卷积+Dense)':<38} | {acc_all*100:<15.2f} | {time_all:<10.2f} | {mem_all:<15.2f}")
    print("="*65)