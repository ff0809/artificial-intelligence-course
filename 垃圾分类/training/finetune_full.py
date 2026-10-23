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

ms.set_seed(42)
np.random.seed(42)

def run_full_tuning_experiment(epochs=100):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", deterministic="ON")
    real_data_dir = "./data/data_en"
    
    print(f"\n运行传统 LeNet5 迁移网络（全量微调）")
    train_loader, num_classes = create_garbage_dataset(real_data_dir, batch_size=32, training=True, model_type="lenet5")
    test_loader, _ = create_garbage_dataset(real_data_dir, batch_size=32, training=False, model_type="lenet5")
    
    net = LeNet5Transfer(num_classes=num_classes, lora_mode="none")
    
    for param in net.get_parameters():
        param.requires_grad = True
        
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-4, weight_decay=1e-4)
    
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
    
    print(f"经典 LeNet5 全量微调测试集最终准确率: {final_acc*100:.2f}% | 总耗时: {time_elapsed:.2f}s | 内存增量: {memory_increase:.2f}MB")
    return final_acc, time_elapsed, memory_increase

if __name__ == "__main__":
    acc_full, time_full, mem_full = run_full_tuning_experiment(epochs=100)
    
    print("\n" + "="*76)
    print("全量微调结果对比")
    print("="*76)
    print(f"{'微调策略范式 (统一基座 LeNet5)':<38} | {'测试集最终准确率(%)':<15} | {'总耗时(s)':<10} | {'内存/显存增量(MB)':<15}")
    print("-" * 76)
    print(f"{'经典全量微调 (Baseline)':<38} | {acc_full*100:<15.2f} | {time_full:<10.2f} | {mem_full:<15.2f}")
    print("="*76)