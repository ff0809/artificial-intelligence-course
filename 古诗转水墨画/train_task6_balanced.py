import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from dataset import ChinesePoemDataset

def main():
    MODEL_FILE_PATH = "/share/home/u15012/paint4poem/v1-5-pruned.safetensors"
    CSV_FILE = "./POEM_IMAGE.csv"
    IMG_DIR = "./images"
    OUTPUT_DIR = "./task6_balanced_checkpoints"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    NUM_SAMPLES = 600

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[任务6-平衡重训练] 设备: {device}, 样本数: {NUM_SAMPLES}, LR: {LEARNING_RATE}")

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_FILE_PATH, torch_dtype=torch.float32,
        safety_checker=None, feature_extractor=None,
        local_files_only=True, low_cpu_mem_usage=True
    )
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler"
    )

    # 双 LoRA
    unet = get_peft_model(unet, LoraConfig(r=8, lora_alpha=16, target_modules=["to_q","to_k","to_v","to_out.0"], lora_dropout=0.0))
    text_encoder = get_peft_model(text_encoder, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.0))
    unet.to(device)
    text_encoder.to(device)

    unet.print_trainable_parameters()
    text_encoder.print_trainable_parameters()

    full_dataset = ChinesePoemDataset(csv_file=CSV_FILE, img_dir=IMG_DIR)
    train_dataset = Subset(full_dataset, range(NUM_SAMPLES))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    params = [p for p in unet.parameters() if p.requires_grad] + [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    accumulation = GRADIENT_ACCUMULATION_STEPS
    for epoch in range(NUM_EPOCHS):
        unet.train()
        text_encoder.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") / accumulation
            loss.backward()

            epoch_loss += loss.item() * accumulation

            if (step+1) % accumulation == 0 or (step+1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if step % (accumulation*5) == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step {step+1}/{len(train_loader)} Loss: {loss.item()*accumulation:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        unet.save_pretrained(os.path.join(epoch_dir, "unet"))
        text_encoder.save_pretrained(os.path.join(epoch_dir, "text_encoder"))

    print("训练完成！")

if __name__ == "__main__":
    main()
