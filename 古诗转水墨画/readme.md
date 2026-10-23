# 基于 Stable Diffusion 的古诗转水墨画生成系统 (Paint4Poem)

本项目为同济大学人工智能课程设计大作业的工程实现，课题为“基于 Stable Diffusion 的古诗转水墨画研究”。

系统包含原始基线推理、Stable Diffusion 全参数微调、LoRA 小样本域迁移、以及条件 GAN 对比基线。

---

## 📂 项目文件组织架构
.
├── dataset.py                      # 任务2/6 水墨画数据集加载与处理脚本
├── run_baseline.py                # 任务1：Stable Diffusion v1.5 单文件基础推理脚本
├── test_baseline.py               # 任务1：命令行参数基线推理脚本
├── train_task2_basic.py            # 任务2：U-Net 全参数微调训练脚本
├── test_task2.py                  # 任务2：U-Net 全参数微调推理脚本
├── train_task3_realistic.py        # 任务3：写实域小样本 LoRA 微调训练脚本
├── test_task3.py                  # 任务3：写实域推理脚本
├── train_task5_cgan_ultimate.py    # 任务5：条件 GAN 训练与推理脚本
├── train_task6_balanced.py         # 任务6：双组件 LoRA 协同微调训练脚本
└── test_task6.py                  # 任务6：动态权重调节推理脚本

## 环境配置与依赖安装

本项目建议使用 GPU 计算环境进行训练与推理。

1. 创建并激活 conda 虚拟环境

```bash
conda create -n paint4poem python=3.10 -y
conda activate paint4poem
```

2. 安装适配 CUDA 11.8 的 PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. 安装 HuggingFace 生态依赖

```bash
pip install diffusers transformers peft accelerate safetensors pandas pillow
```

### 🗃️ 基础模型权重准备

请准备 Stable Diffusion v1.5 剪裁版单文件模型权重 `v1-5-pruned.safetensors`，并放在服务器可访问路径中。

## 数据集架构与配置

### 1. 核心古诗水墨画数据集 (`Paint4Poem`)

* **索引表格**：`POEM_IMAGE.csv` 必须使用 **制表符 (`\t`)** 进行列切分，列名应规范清理为 `image_id` 与 `poem`。


* **图像资产**：存放在 `./images` 路径下。`ChinesePoemDataset` 内部已封装多扩展名自适应搜索逻辑（`.jpg`, `.png`, `.jpeg` 等），可兼容各种图像格式。



### 2. 自建拓展测试写实数据集 (任务 3)

* **索引表格**：`REALISTIC_IMAGE.csv` 必须使用 **逗号 (`,`)** 进行标准列切分。


* **图像资产**：存放在 `../data/realistic_images` 路径下，用于承载异质域小样本微调实验。

## 核心任务全流程运行指南

### 任务 1：部署原生底模执行基线测试

直接调用离线单文件底模，基于固定种子生成宏观或特定主体的水墨画样本，用于暴露原生大模型由于跨语言分词导致的中文“语义断裂”硬伤。

```bash
python test_baseline.py \
  --prompt "Lonely boat, old fisherman in a straw raincoat, fishing alone on a frozen river with heavy snow, traditional Chinese ink wash painting, high quality" \
  --steps 30 \
  --guidance 7.5 \
  --seed 42 \
  --output "./baseline_out.png"

```

### 任务 2：U-Net 空间主干全参数微调

全参数微调范式下，锁定 VAE 与 Text Encoder 不参与反向传播，彻底解冻 U-Net 的全部 8.6 亿参数参与梯度更新。

#### 1. 启动全参数反向扩散微调

脚本内部激活了梯度检查点（Gradient Checkpointing）机制，并配合大跨度的梯度累积，将高密度的全参数微调开销成功压缩进常规显存中。

```bash
# 内部默认超参：BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=8 (等效总Batch=32)
# LEARNING_RATE=5e-6 (极小学习率防遗忘), NUM_EPOCHS=30, 样本数=600
python train_task2_basic.py

```

微调生成的完整 U-Net 权重自动固化保存在 `./task2_basic_checkpoints_full/epoch_30/unet` 目录下。

#### 2. 调用全参数模型进行定向推理

```bash
python test_task2.py \
  --prompt "孤舟蓑笠翁，独钓寒江雪。Lonely boat, old fisherman fishing on snowy river, traditional Chinese ink wash painting, 水墨画风格" \
  --output "./task2_out.png"

```

---

### 任务 3：自建写实域小样本风格迁移

该任务使用 LoRA 进行写实风格的域迁移训练。

#### 1. 启动写实小样本微调

```bash
python train_task3_realistic.py
```

适配器权重保存在 `./task3_realistic_checkpoints/epoch_40`。

#### 2. 执行推理

```bash
python test_task3.py \
  --prompt "Lonely sail on the Yangtze River, realistic photography style, 写实风格，高清照片" \
  --output "./task3_out.png"
```

---

### 任务 5：条件 GAN 基线

该任务使用纯 PyTorch 条件 GAN 作为对比基线。

```bash
python train_task5_cgan_ultimate.py
```

---

### 任务 6：双组件 LoRA 协同训练与推理

该任务对 U-Net 和文本编码器同时进行 LoRA 微调。

#### 1. 启动双通道 LoRA 协同微调

```bash
python train_task6_balanced.py
```

轻量适配器模型保存在 `./task6_balanced_checkpoints/epoch_30`。

#### 2. 推理期动态权重调节

```bash
python test_task6.py \
  --prompt "孤帆远影碧空尽，唯见长江天际流。Lonely sail fades into blue sky, only the Yangtze River flows to the horizon, ink wash painting, 水墨画风格" \
  --unet_scale 0.7 \
  --text_scale 0.2 \
  --seed 42 \
  --output "./task6_out.png"
```

---

## 理想效果与提示词工程（Prompt Engineering）指南

为了在项目验收和结题成果中最大化激发模型的隐空间表现，建议遵循以下中英双向跨模态锚定（Bilingual Anchoring）格式输入提示词：

> **“【中文诗句名句】 + 【英文具象空间实体与几何构图描述】 + traditional Chinese ink wash painting, 水墨画风格, 留白, 皴法”**

### 黄金测试提示词矩阵推荐

1. **山水深远意境（推荐用于测试任务 6 动态路由）**：
`“两岸青山相对出，孤帆一片日边来。Two green mountains face each other, a lonely sail comes from the sun, traditional Chinese ink wash painting, 水墨画风格”`
2. **寒江极简写意（推荐用于测试任务 2 / 任务 6 材质对比）**：
`“孤舟蓑笠翁，独钓寒江雪。Lonely boat, old fisherman fishing on snowy river, traditional Chinese ink wash painting, 水墨画风格”`
3. **大漠古宣纸底色触发器**：
`“白日依山尽，黄河入海流。The sun sets behind mountains, the Yellow River flows into the sea, ancient Chinese masterpiece style, aged yellow paper texture, deep ink strokes”`

```

```