import torch
import time
from PIL import Image, ImageDraw
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

def create_mock_sketch():
    """用代码绘制一个简单的五官线条图，模拟前端传来的不完整草图"""
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    # 画一个非常抽象的动漫脸部轮廓和眼睛（黑线）
    draw.ellipse([120, 100, 390, 420], outline="black", width=3) # 脸型
    draw.ellipse([180, 210, 230, 250], outline="black", width=4) # 左眼
    draw.ellipse([280, 210, 330, 250], outline="black", width=4) # 右眼
    draw.arc([210, 310, 300, 350], start=0, end=180, fill="black", width=4) # 嘴巴
    return img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在配置设备: {device} (RTX 4080 准备就绪)...")

    # 1. 加载轻量化的二次元底座和标准 Scribble ControlNet
    print("正在从库中加载基础模型与 ControlNet (初次加载需要下载，请保持网络畅通)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )
    
    # 使用社区优秀的二次元模型 Counterfeit-V3.0
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stablediffusionapi/counterfeit-v30", 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = None

    # 2. 包装进 StreamDiffusion 流式流水线
    # t_index_list=[0, 1, 2, 3] 代表只走 4 步快速采样，这是毫秒级响应的核心机制
    stream = StreamDiffusion(
        pipe,
        t_index_list=[0, 1, 2, 3],
        torch_dtype=torch.float16,
    )

    # 3. 预热引擎与生成条件注入
    prompt = "1girl, anime portrait, high quality, masterpiece, solo, colorful"
    stream.prepare(prompt=prompt, num_inference_steps=4)
    print("✨ 模型预热完毕！开始进行实时绘图模拟测试...")

    # 4. 模拟用户画画，连续发送 10 次增量笔画的渲染循环
    mock_sketch = create_mock_sketch()
    
    for i in range(10):
        start_time = time.time()
        
        # 将草图送入流式管道
        x_output = stream(mock_sketch)
        
        # 后处理渲染成彩色 PIL 图像
        output_image = postprocess_image(x_output, output_type="pil")[0]
        
        latency = (time.time() - start_time) * 1000
        print(f"第 {i+1} 笔刷新成功 | 本帧延迟 (Latency): {latency:.2f} ms")
        
        # 把最后一笔生成的全彩成品保存下来
        if i == 9:
            output_image.save("final_anime_guidance.png")
            print("\n🎉 测试成功完成！")
            print("已经根据你的抽象草图，脑补出了全彩动漫头像！")
            print("图片已保存至当前目录：final_anime_guidance.png")

if __name__ == "__main__":
    main()