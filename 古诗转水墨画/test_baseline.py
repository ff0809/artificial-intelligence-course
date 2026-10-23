import argparse
import torch
from diffusers import StableDiffusionPipeline

def main():
    parser = argparse.ArgumentParser(description="任务1：原始SD模型推理")
    parser.add_argument("--prompt", type=str, required=True, help="提示词")
    parser.add_argument("--steps", type=int, default=30, help="去噪步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="引导系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="./baseline_out.png", help="输出路径")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/path/to/v1-5-pruned.safetensors"  

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH, torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True
    ).to(device)

    generator = torch.Generator(device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=generator).images[0]
    image.save(args.output)
    print(f"图片已保存: {args.output}")

if __name__ == "__main__":
    main()