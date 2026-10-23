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

def run_freeze_experiment(strategy="freeze_base", epochs=100):
    ms.set_seed(42)
    np.random.seed(42)
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", deterministic="ON")
    real_data_dir = "./data/data_en" 
    
    train_loader, num_classes = create_garbage_dataset(real_data_dir, batch_size=32, training=True, model_type="lenet5")
    test_loader, _ = create_garbage_dataset(real_data_dir, batch_size=32, training=False, model_type="lenet5")
    
    net = LeNet5Transfer(num_classes=num_classes, lora_mode="none")
    
    if strategy == "freeze_base":
        print(f"\n仅开放最后一层全连接分类头 (fc3_garbage)")
        for param in net.get_parameters():
            if "fc3_garbage" in param.name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif strategy == "freeze_fc2_fc3":
        print(f"\n仅开放最后两层全连接 (fc2 + fc3_garbage)")
        for param in net.get_parameters():
            if "backbone.6" in param.name or "fc3_garbage" in param.name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif strategy == "freeze_improved":
        print(f"\n解放全部全连接决策层 (fc1 + fc2 + fc3_garbage)")
        for param in net.get_parameters():
            if "backbone.5" in param.name or "backbone.6" in param.name or "fc3_garbage" in param.name:
                param.requires_grad = True  
            else:
                param.requires_grad = False 
                
    elif strategy == "freeze_conv1_only":
        print(f"\n仅冻结最初始的第一层卷积 (conv1)，全面解放后续架构")
        for param in net.get_parameters():
            if "backbone.0" in param.name: 
                param.requires_grad = False 
            else:
                param.requires_grad = True  

    if strategy == "freeze_base":
        learning_rate = 1e-3
    else:
        learning_rate = 1e-4  

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate, weight_decay=5e-4)
    
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
    
    print(f"策略: {strategy} | 测试集最终准确率: {final_acc*100:.2f}% | 耗时: {time_elapsed:.2f}s | 内存增量: {memory_increase:.2f}MB")
    return final_acc, time_elapsed, memory_increase

if __name__ == "__main__":
    acc_base, time_base, mem_base = run_freeze_experiment(strategy="freeze_base", epochs=100)
    acc_fc2, time_fc2, mem_fc2 = run_freeze_experiment(strategy="freeze_fc2_fc3", epochs=100)
    acc_imp, time_imp, mem_imp = run_freeze_experiment(strategy="freeze_improved", epochs=100)
    acc_conv1, time_conv1, mem_conv1 = run_freeze_experiment(strategy="freeze_conv1_only", epochs=100)
    
    print("\n" + "="*80)
    print("四种冻结策略对比")
    print("="*80)
    print(f"{'冻结微调范围策略 (统一基座 LeNet5)':<38} | {'测试集最终准确率(%)':<15} | {'总耗时(s)':<10} | {'内存/显存增量(MB)':<15}")
    print("-" * 80)
    print(f"{'基础冻结 (仅 fc3 开放微调)':<38} | {acc_base*100:<15.2f} | {time_base:<10.2f} | {mem_base:<15.2f}")
    print(f"{'渐进半冻结 (fc2 + fc3 开放微调)':<38} | {acc_fc2*100:<15.2f} | {time_fc2:<10.2f} | {mem_fc2:<15.2f}")
    print(f"{'高级半冻结 (fc1 + fc2 + fc3 开放微调)':<38} | {acc_imp*100:<15.2f} | {time_imp:<10.2f} | {mem_imp:<15.2f}")
    print(f"{'浅层微调 (仅冻结第一层 conv1，其余全开放)':<38} | {acc_conv1*100:<15.2f} | {time_conv1:<10.2f} | {mem_conv1:<15.2f}")
    print("="*80)