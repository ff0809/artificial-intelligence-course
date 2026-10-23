import argparse
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def main():
    parser = argparse.ArgumentParser(description="任务2：全参数微调模型推理")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./task2_out.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/path/to/v1-5-pruned.safetensors"
    UNET_PATH = "./task2_basic_checkpoints_full/epoch_30/unet"  

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH, torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True
    )
    fine_tuned_unet = UNet2DConditionModel.from_pretrained(UNET_PATH, torch_dtype=torch.float16)
    pipe.unet = fine_tuned_unet
    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=generator).images[0]
    image.save(args.output)
    print(f"图片已保存: {args.output}")

if __name__ == "__main__":
    main()