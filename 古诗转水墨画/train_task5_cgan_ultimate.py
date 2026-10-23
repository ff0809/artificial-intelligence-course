import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer

class BilingualGANDataset(Dataset):
     def __init__(self, csv_file, img_dir, size=64):
         self.df = pd.read_csv(csv_file, sep='\t')
         self.df.columns = [c.strip() for c in self.df.columns]
         self.img_dir = img_dir
         
         self.tokenizer = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
         self.transform = transforms.Compose([
             transforms.Resize((size, size)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

     def __len__(self):
         return min(len(self.df), 2500)

     def __getitem__(self, idx):
         row = self.df.iloc[idx]
         img_id = str(row['image_id']).strip()
         img_path = None
         for ext in ['.jpeg', '.jpg', '.png', '.JPG', '.JPEG']:
             candidate = os.path.join(self.img_dir, f"{img_id}{ext}")
             if os.path.exists(candidate):
                 img_path = candidate
                 break
         try:
             image = Image.open(img_path).convert("RGB")
             image = self.transform(image)
         except:
             image = torch.zeros(3, 64, 64)
             
         inputs = self.tokenizer(str(row['poem']), max_length=77, padding="max_length", truncation=True, return_tensors="pt")
         return image, inputs.input_ids.squeeze(0)

class TextEncoder(nn.Module):
     def __init__(self, vocab_size=49408, embed_dim=64, hidden_dim=64):
         super().__init__()
         self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
         self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
     def forward(self, text):
         x = self.embed(text)
         _, h = self.gru(x)
         return h[-1]  

class StableConditionalGenerator(nn.Module):
     def __init__(self, z_dim=100, text_dim=64, img_ch=3):
         super().__init__()
         # 引入非线性条件映射，防止文本噪音直接散布
         self.text_layer = nn.Sequential(
             nn.Linear(text_dim, 64),
             nn.LeakyReLU(0.2, inplace=True)
         )
         self.fc = nn.Linear(z_dim + 64, 512 * 4 * 4)
         self.upsample = nn.Sequential(
             nn.BatchNorm2d(512), nn.ReLU(True),
             # 4x4 -> 8x8
             nn.Upsample(scale_factor=2, mode='nearest'),
             nn.Conv2d(512, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
             # 8x8 -> 16x16
             nn.Upsample(scale_factor=2, mode='nearest'),
             nn.Conv2d(256, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
             # 16x16 -> 32x32
             nn.Upsample(scale_factor=2, mode='nearest'),
             nn.Conv2d(128, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
             # 32x32 -> 64x64
             nn.Upsample(scale_factor=2, mode='nearest'),
             nn.Conv2d(64, img_ch, 3, 1, 1, bias=False), 
             nn.Tanh()
         )
     def forward(self, noise, text_embed):
         t_feat = self.text_layer(text_embed)
         flatten_noise = noise.view(noise.size(0), -1)
         x = torch.cat([flatten_noise, t_feat], dim=1)
         x = self.fc(x).view(-1, 512, 4, 4)
         return self.upsample(x)

class SpectralDiscriminator(nn.Module):
     def __init__(self, img_ch=3, text_dim=64):
         super().__init__()
         self.text_proj = nn.Linear(text_dim, 64 * 64)
         
         self.conv1 = nn.utils.spectral_norm(nn.Conv2d(img_ch + 1, 64, 4, 2, 1, bias=False))
         self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False))
         self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False))
         self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False))
         self.final = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
         self.leaky = nn.LeakyReLU(0.2, inplace=True)
         
     def forward(self, img, text_embed):
         t_map = self.text_proj(text_embed).view(-1, 1, 64, 64)
         x = torch.cat([img, t_map], dim=1)
         
         x = self.leaky(self.conv1(x))
         x = self.leaky(self.conv2(x))
         x = self.leaky(self.conv3(x))
         x = self.leaky(self.conv4(x))
         return torch.sigmoid(self.final(x)).view(-1)

def main():
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print(f"[任务5谱归一化防爆版] 设备: {device}")

     dataset = BilingualGANDataset("./POEM_IMAGE.csv", "./images")
     dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

     text_enc = TextEncoder().to(device)
     netG = StableConditionalGenerator().to(device)
     netD = SpectralDiscriminator().to(device)

     criterion = nn.BCELoss()
     
     # TTUR双时间尺度更新规则：让生成器学习率稍快，判别器稍慢，维持动态平衡
     optG = optim.Adam(list(netG.parameters()) + list(text_enc.parameters()), lr=0.0003, betas=(0.5, 0.999))
     optD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))

     epochs = 40
     print("\n谱平衡数据大盘微调正式启动...")
     for epoch in range(epochs):
          for real_imgs, texts in dataloader:
              batch = real_imgs.size(0)
              real_imgs = real_imgs.to(device)
              texts = texts.to(device)

              # 训练 D
              with torch.no_grad():
                  text_embed_d = text_enc(texts).detach()
              noise = torch.randn(batch, 100, 1, 1, device=device)
              fake = netG(noise, text_embed_d).detach()

              real_labels = torch.full((batch,), 0.9, device=device) # 软化标签
              fake_labels = torch.full((batch,), 0.1, device=device)

              optD.zero_grad()
              out_real = netD(real_imgs, text_embed_d)
              loss_real = criterion(out_real, real_labels)
              out_fake = netD(fake, text_embed_d)
              loss_fake = criterion(out_fake, fake_labels)
              lossD = loss_real + loss_fake
              lossD.backward()
              optD.step()

              # 训练 G
              text_embed_g = text_enc(texts)  
              fake_g = netG(noise, text_embed_g)
              out_g = netD(fake_g, text_embed_g)
              lossG = criterion(out_g, torch.full((batch,), 1.0, device=device))

              optG.zero_grad()
              lossG.backward()
              optG.step()

          if (epoch+1) % 5 == 0 or epoch == 0:
              print(f"   Epoch [{epoch+1}/{epochs}] D_loss: {lossD.item():.4f} G_loss: {lossG.item():.4f}")

     target_poem = "lonely boat, old fisherman fishing on frozen river, ink wash painting, 水墨画风格"
     print(f"\n[谱归一化现场推理] 驱动目标 ➔ 「{target_poem}」")
     
     netG.eval()
     text_enc.eval()
     with torch.no_grad():
          text_tensor = dataset.tokenizer(target_poem, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
          text_embed = text_enc(text_tensor)
          noise = torch.randn(1, 100, 1, 1, device=device)
          fake_img = netG(noise, text_embed)
          
     fake_img = (fake_img + 1) / 2.0
     ndarr = fake_img.squeeze().permute(1, 2, 0).cpu().numpy()

     img_pil = Image.fromarray((ndarr * 255).astype('uint8')).convert("L").convert("RGB")

     pixels = img_pil.load()
     for y in range(img_pil.size[1]):
         for x in range(img_pil.size[0]):
             r, g, b = pixels[x, y]
             pixels[x, y] = (int(r * 1.0), int(g * 0.92), int(b * 0.82))
             
     img_pil.save("./Task5_Alternative_cGAN_Result.png")
     print("图片已保存在: Task5_Alternative_cGAN_Result.png")

if __name__ == "__main__":
     main()