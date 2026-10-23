# garbage_dataset.py
import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as C

def create_garbage_dataset(data_dir, batch_size=32, training=True, model_type="lenet5"):
    split = "train" if training else "test"
    target_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"找不到对应的垃圾分类数据集路径: {target_dir}，请检查路径配置！")
        
    classes = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    dataset = ds.ImageFolderDataset(dataset_dir=target_dir, 
                                    decode=False, 
                                    class_indexing=class_to_idx,
                                    shuffle=training)
    
    transform = [vision.Decode()]
    
    if model_type == "lenet5":
        transform += [
            vision.ConvertColor(vision.ConvertMode.COLOR_RGB2GRAY),
            lambda img: np.expand_dims(img, axis=-1), 
            vision.Resize((32, 32)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=[0.1307], std=[0.3081]),
            vision.HWC2CHW()
        ]
    else:
        if training:
            transform += [vision.RandomHorizontalFlip(prob=0.5)]
            
        transform += [
            vision.Resize((224, 224)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]

    dataset = dataset.map(transform, input_columns="image")
    dataset = dataset.map(C.TypeCast(ms.int32), input_columns="label")
    dataset = dataset.batch(batch_size, drop_remainder=training)
    return dataset, len(classes)