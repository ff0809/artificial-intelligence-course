import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer

class RealisticDataset(Dataset):
    def __init__(self, csv_file, img_dir, size=512):
        self.df = pd.read_csv(csv_file, sep=',')   # 逗号分隔(与paint4poem不同)
        self.img_dir = img_dir
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer"
        )
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['image_id']).strip()
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (512, 512), (0, 0, 0))
        pixel_values = self.transform(image)
        inputs = self.tokenizer(
            str(row['poem']),
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids.squeeze(0)}

def main():
    MODEL_FILE_PATH = "/path/to/v1-5-pruned.safetensors" 
    CSV_FILE = "../data/REALISTIC_IMAGE.csv"
    IMG_DIR = "../data/realistic_images"
    OUTPUT_DIR = "./task3_realistic_checkpoints"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"任务3: 写实微调, 设备 {device}")

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_FILE_PATH, torch_dtype=torch.float32,
        safety_checker=None, local_files_only=True
    )
    unet, text_encoder, vae = pipe.unet, pipe.text_encoder, pipe.vae
    vae.requires_grad_(False)
    vae.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler"
    )

    unet = get_peft_model(unet, LoraConfig(r=8, lora_alpha=16,
                          target_modules=["to_q","to_k","to_v","to_out.0"]))
    text_encoder = get_peft_model(text_encoder, LoraConfig(r=8, lora_alpha=16,
                                 target_modules=["q_proj","v_proj"]))
    unet.to(device)
    text_encoder.to(device)

    dataset = RealisticDataset(csv_file=CSV_FILE, img_dir=IMG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = [p for p in unet.parameters() if p.requires_grad] + \
             [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        unet.train()
        text_encoder.train()
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                     (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} 平均损失: {epoch_loss/len(loader):.4f}")
        if (epoch+1) == NUM_EPOCHS:
            epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            unet.save_pretrained(os.path.join(epoch_dir, "unet"))
            text_encoder.save_pretrained(os.path.join(epoch_dir, "text_encoder"))
            print(f"权重已保存至 {epoch_dir}")

if __name__ == "__main__":
    main()