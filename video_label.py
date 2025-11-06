from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
import os
from datetime import datetime

def load_prompts():
    """加载prompts配置文件"""
    with open('prompts.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_objects():
    """加载物体名词配置文件"""
    with open('objects_list.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_model_and_processor():
    """
    设置模型和处理器，配置视觉token预算
    """
    # 加载模型
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", 
        dtype="auto", 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    
    # 配置图像处理器的视觉token预算
    processor.image_processor.size = {
        "longest_edge": 1280 * 32 * 32,
        "shortest_edge": 256 * 32 * 32
    }
    
    # 配置视频处理器的视觉token预算  
    processor.video_processor.size = {
        "longest_edge": 16384 * 32 * 32,
        "shortest_edge": 256 * 32 * 32
    }
    
    return model, processor

def create_phase_analysis_prompt(prompt_key="phase_analysis", operation_type="overall", custom_objects=None):
    """从配置文件加载prompt"""
    # 加载prompts配置
    prompts_config = load_prompts()
    
    if prompt_key not in prompts_config:
        available_prompts = list(prompts_config.keys())
        raise ValueError(f"Prompt '{prompt_key}' not found. Available prompts: {available_prompts}")
    
    # 加载物体名词
    objects_config = load_objects()
    # 修复：直接使用 get 方法，如果不存在则返回 overall 类别
    print(objects_config)
    print(operation_type)
    base_objects = objects_config.get(operation_type, objects_config.get("overall", []))
    
    # 如果 overall 也不存在，使用空列表
    if not base_objects:
        base_objects = []
        print("警告: 配置文件中未找到物体列表")
    
    # 合并物体列表
    objects_list = base_objects.copy()
    if custom_objects:
        objects_list.extend(custom_objects)
    
    objects_text = ", ".join(objects_list)
    
    # 获取选中的prompt并填充物体文本
    selected_prompt = prompts_config[prompt_key]["prompt"]
    prompt = selected_prompt.format(objects_text=objects_text)
    
    print(f"使用prompt: {prompts_config[prompt_key]['name']}")
    print(f"操作类型: {operation_type}")
    print(f"物体数量: {len(objects_list)}")
    
    return prompt

def save_phase_results(result_text, video_name, output_dir="phase_results"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{video_name}_phases_{timestamp}.json"  # 改为.json后缀
    filepath = os.path.join(output_dir, filename)
    
    try:
        import json
        # 清理可能的多余字符，提取JSON部分
        json_str = result_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].strip()
        

        phase_data = json.loads(json_str)
        
        phase_data["metadata"] = {
            "video_file": video_name,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(phase_data, f, ensure_ascii=False, indent=2)
        
        print(f"Analysis results saved to: {filepath}")
        return filepath
        
    except json.JSONDecodeError as e:
        # 如果JSON解析失败，保存原始文本
        print(f"JSON parsing failed, saving as text: {e}")
        txt_filepath = filepath.replace('.json', '.txt')
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write("Robot Operation Phase Analysis Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Video File: {video_name}\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result_text)
        return txt_filepath
    
def analyze_operation_phases(
    video_path, 
    prompt_key="phase_analysis",
    operation_type="overall",
    custom_objects=None,
    fps=8,  
    max_new_tokens=1024
):

    model, processor = setup_model_and_processor()
    
    prompt = create_phase_analysis_prompt(prompt_key, operation_type, custom_objects)
    

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


    processing_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "fps": fps
    }

    inputs = processor.apply_chat_template(
        messages,
        **processing_kwargs
    )
    inputs = inputs.to(model.device)


    print("正在分析操作阶段...")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def batch_analyze_videos(
    video_folder,
    prompt_key="phase_analysis",
    operation_type="overall", 
    custom_objects=None,
    fps=8,
    max_new_tokens=1024,
    supported_formats=['.mp4', '.avi', '.mov', '.mkv', '.webm']
):
    """
    批量分析文件夹内的视频文件
    """
    # 一次性加载模型，避免重复加载
    print("正在加载模型和处理器...")
    model, processor = setup_model_and_processor()
    print("模型加载完成！")
    
    # 获取视频文件列表
    video_files = []
    for file in os.listdir(video_folder):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in supported_formats:
            video_files.append(os.path.join(video_folder, file))
    
    if not video_files:
        print(f"在文件夹 '{video_folder}' 中未找到支持的视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for video_file in video_files:
        print(f"  - {os.path.basename(video_file)}")
    
    # 批量处理
    results = {}
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\n{'='*60}")
        print(f"处理进度: {i}/{len(video_files)} - {video_name}")
        print(f"{'='*60}")
        
        try:
            # 复用已加载的模型和处理器
            prompt = create_phase_analysis_prompt(prompt_key, operation_type, custom_objects)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            processing_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
                "fps": fps
            }

            inputs = processor.apply_chat_template(messages, **processing_kwargs)
            inputs = inputs.to(model.device)

            print("正在分析操作阶段...")
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            result = output_text[0]
            results[video_name] = result
            
            # 保存结果
            save_phase_results(result, video_name)
            print(f"✓ 完成: {video_name}")
            
        except Exception as e:
            print(f"✗ 处理失败 {video_name}: {e}")
            results[video_name] = f"Error: {e}"
    
    print(f"\n批量处理完成！共处理 {len(video_files)} 个视频")
    return results

if __name__ == "__main__":
    # video_path = "./video/task_agnostic_open_drawer.mp4"  
    
    # print("=== 机器人操作阶段分析 ===")
    # result = analyze_operation_phases(
    #     video_path=video_path,
    #     prompt_key="phase_analysis",
    #     operation_type="overall",  
    #     custom_objects=["tool", "towel"],  # 可添加自定义物体，不填也可以
    #     fps=10 # 自定义采样帧率，精细操作需要更高采样率
    # )
    
    # print("分析结果:")
    # print("=" * 50)
    # print(result)
    

    # video_name = os.path.basename(video_path)
    # save_phase_results(result, video_name)

    video_folder = "./video"  # 视频文件夹路径
    print("=== 批量机器人操作阶段分析 ===")
    batch_results = batch_analyze_videos(
        video_folder=video_folder,
        prompt_key="phase_analysis",
        operation_type="overall",
        custom_objects=["tool", "towel"],
        fps=10
    )