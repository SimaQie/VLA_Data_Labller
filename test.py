from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from PIL import Image
import os

# 默认：在可用设备上加载模型
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)


processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

def analyze_local_image(image_path, prompt="Describe this image."):
    """
    分析本地图片的函数
    
    Args:
        image_path (str): 本地图片路径
        prompt (str): 自定义的提示词
    """
    if not os.path.exists(image_path):
        print(f"错误：图片文件 '{image_path}' 不存在")
        return
    
    try:
        image = Image.open(image_path)
        print(f"成功加载图片: {image_path}")
        print(f"图片尺寸: {image.size}")
        print(f"图片模式: {image.mode}")
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # 直接使用PIL Image对象
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 准备推理
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # 推理：生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("\n" + "="*50)
        print("分析结果:")
        print("="*50)
        print(output_text[0])
        
    except Exception as e:
        print(f"处理图片时出错: {e}")

# 使用示例
if __name__ == "__main__":
    image_path = "./images/grab_plate_1.jpg"  # 替换为你的图片路径
    prompt = "Describe the objects in this scenario."
    analyze_local_image(image_path, prompt)
    
 
    # while True:
    #     print("\n" + "="*50)
    #     user_image_path = input("请输入图片路径（输入'quit'退出）: ")
        
    #     if user_image_path.lower() == 'quit':
    #         break
            
    #     user_prompt = input("请输入分析提示词（直接回车使用默认提示词）: ")
    #     if not user_prompt.strip():
    #         user_prompt = "Describe this image."
            
    #     analyze_local_image(user_image_path, user_prompt)