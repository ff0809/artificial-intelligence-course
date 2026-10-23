import torch
from diffusers import StableDiffusionPipeline

MODEL_FILE_PATH = r"D:\人工智能原理与技术\古诗转水墨画\v1-5-pruned.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    MODEL_FILE_PATH,
    torch_dtype=torch.float16,
    load_safety_checker=False  # 关闭安全检查器以加快加载速度并节省显存
)

pipe = pipe.to("cuda")

prompt = "A traditional Chinese ink wash painting of mountains and a river, foggy weather, high quality"

with torch.inference_mode():
    output = pipe(
        prompt=prompt,
        num_inference_steps=30,  # 迭代去噪步数
        guidance_scale=7.5       # 提示词引导系数
    )

image = output.images[0]
image.save("./my_baseline_result.png")

print("运行成功！生成的基线图片已保存!")