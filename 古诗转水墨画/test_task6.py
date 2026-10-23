import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def set_lora_scale(model, adapter_name, scale_factor):
    """手动设置 LoRA 缩放系数，兼容不同 peft 版本"""
    if adapter_name in model.peft_config:
        config = model.peft_config[adapter_name]
        base_scaling = config.lora_alpha / config.r
        for module in model.modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                if adapter_name in module.scaling:
                    module.scaling[adapter_name] = base_scaling * scale_factor

def main():
    parser = argparse.ArgumentParser(description="任务6：双LoRA模型推理（支持动态权重）")
    parser.add_argument("--prompt", type=str, required=True, help="中文古诗或中英文混合提示词")
    parser.add_argument("--unet_scale", type=float, default=0.7, help="U-Net LoRA 权重 (0-1)")
    parser.add_argument("--text_scale", type=float, default=0.2, help="Text Encoder LoRA 权重 (0-1)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./task6_out.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/path/to/v1-5-pruned.safetensors"
    LORA_PATH = "./task6_balanced_checkpoints/epoch_30"  

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH, torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(LORA_PATH, "unet"), adapter_name="unet_lora")
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, os.path.join(LORA_PATH, "text_encoder"), adapter_name="text_lora")
    pipe = pipe.to(device)

    set_lora_scale(pipe.unet, "unet_lora", args.unet_scale)
    set_lora_scale(pipe.text_encoder, "text_lora", args.text_scale)

    print(f"提示词: {args.prompt}")
    print(f"权重: U-Net={args.unet_scale}, Text={args.text_scale}, 种子={args.seed}")

    generator = torch.Generator(device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=generator).images[0]
    image.save(args.output)
    print(f"图片已保存: {args.output}")

if __name__ == "__main__":
    main()