import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="任务3：写实风格LoRA模型推理")
    parser.add_argument("--prompt", type=str, required=True, help="英文描述")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./task3_out.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/path/to/v1-5-pruned.safetensors"
    LORA_PATH = "./task3_realistic_checkpoints/epoch_40" 

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH, torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(LORA_PATH, "unet"))
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(LORA_PATH, "text_encoder"))
    pipe.unet = pipe.unet.merge_and_unload()
    pipe.text_encoder = pipe.text_encoder.merge_and_unload()
    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=generator).images[0]
    image.save(args.output)
    print(f"图片已保存: {args.output}")

if __name__ == "__main__":
    main()