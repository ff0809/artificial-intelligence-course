import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

class ChinesePoemDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer_id="stable-diffusion-v1-5/stable-diffusion-v1-5", size=512):
        self.df = pd.read_csv(csv_file, sep="\t")
        self.img_dir = img_dir
        
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_id, subfolder="tokenizer")
        
        self.image_transforms = transforms.Compose([
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
        
        possible_names = [
            f"{img_id}.jpg", f"{img_id}.jpeg", f"{img_id}.png",
            f"{img_id}.JPG", f"{img_id}.JPEG", f"{img_id}.PNG",
            f"img_{img_id}.jpg", f"img_{img_id}.jpeg", f"img_{img_id}.png"
        ]
        
        img_path = None
        for name in possible_names:
            test_path = os.path.join(self.img_dir, name)
            if os.path.exists(test_path):
                img_path = test_path
                break

        try:
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                raise FileNotFoundError
        except Exception:
            image = Image.new("RGB", (512, 512), (0, 0, 0))
            
        pixel_values = self.image_transforms(image)
        poem_text = str(row['poem'])
        
        inputs = self.tokenizer(
            poem_text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
            "poem": poem_text
        }
