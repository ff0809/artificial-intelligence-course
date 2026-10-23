import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from diffusers import StableDiffusionPipeline, DDPMScheduler

from dataset import ChinesePoemDataset

def main():
    MODEL_FILE_PATH = "/share/home/u15012/paint4poem/v1-5-pruned.safetensors"
    CSV_FILE = "./POEM_IMAGE.csv"
    IMG_DIR = "./images"
    OUTPUT_DIR = "./task2_basic_checkpoints_full"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 4                     # 单次喂入4张图
    GRADIENT_ACCUMULATION_STEPS = 8    # 累积8步，有效batch = 32
    LEARNING_RATE = 5e-6               # 全参数学习率必须很小
    NUM_EPOCHS = 30
    NUM_SAMPLES = 600                  # 只用前600张图片

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[任务2-纯正全参数微调] 运行设备: {device}")
    print(f"使用样本数: {NUM_SAMPLES}, 单步batch: {BATCH_SIZE}, 梯度累积步: {GRADIENT_ACCUMULATION_STEPS}")

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_FILE_PATH,
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    # 冻结 VAE 和文本编码器
    vae.requires_grad_(False)
    vae.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    # 解冻整个 U-Net（全参数微调）
    unet.requires_grad_(True)
    unet.to(device)

    # 开启梯度检查点，大幅降低显存占用
    unet.enable_gradient_checkpointing()

    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"🧬 任务(2) 全参数微调模式：可训练参数量 = {trainable_params:,} (约占 U-Net 总参数的 100%)")

    # 调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler"
    )

    full_dataset = ChinesePoemDataset(csv_file=CSV_FILE, img_dir=IMG_DIR)
    train_dataset = Subset(full_dataset, range(NUM_SAMPLES))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 优化器（优化所有 U-Net 参数）
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    print("启动任务（2）全参数反向扩散微调循环（无 LoRA）...")
    accumulation_steps = GRADIENT_ACCUMULATION_STEPS

    for epoch in range(NUM_EPOCHS):
        unet.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # 梯度累积：损失除以累积步数
            loss = loss / accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

            if step % (accumulation_steps * 2) == 0:  
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step+1}/{len(train_dataloader)}] - 实时 Loss: {loss.item() * accumulation_steps:.4f}")

        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        unet.save_pretrained(os.path.join(epoch_dir, "unet"))
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"📊 Epoch {epoch+1} 全参数微调完毕，本轮平均 Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
