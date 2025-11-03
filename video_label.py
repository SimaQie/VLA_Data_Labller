from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
import os
from datetime import datetime

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

def create_phase_analysis_prompt(operation_type="general", custom_objects=None):

    objects_list = get_object_vocabulary(operation_type)
    if custom_objects:
        objects_list.extend(custom_objects)
    
    objects_text = ", ".join(objects_list)
    
    prompt = f"""
You are a trainer explaining a robot's operation to a new technician. Watch the video and generate a step-by-step guide.",
For each key step in the operation, identify the exact moment in the video where that step begins. Focus on moments where the robot's primary action changes."

Objects that may appear in the video include: {objects_text}

Please output the analysis results in JSON format with the following structure:

{{
    "phases": [
        {{
            "phase_number": 1,
            "phase_name": "phase name",
            "start_time": "0:00",
            "end_time": "0:15",
            "description": "Detailed description of the operation content, robot actionsand objects used in this phase"
        }},
        {{
            "phase_number": 2,
            "phase_name": "phase name", 
            "start_time": "0:15",
            "end_time": "0:45",
            "description": "Detailed description..."
        }}
    ],
    "summary": "Brief overall summary of the operation"
}}

Requirements:
1. Time format must use "minutes:seconds", e.g., "0:15-0:45"
2. Phase names should be concise and clear
3. Descriptions should be specific, including objects used and specific actions
4. Ensure temporal continuity without overlaps
5. Output ONLY valid JSON, no additional text
"""
    return prompt

def get_object_vocabulary(operation_type="general"):
    """
    Function 2: Noun vocabulary containing possible objects (English version)
    """
    vocabularies = [
            "robot", "robotic arm", "end effector", "plate", "cup", "rack", "shelf",
            "table", "box", "container", "bowl", "screwdriver", "gripper", "clothes"
        ]
 
    return vocabularies

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
    operation_type="general",
    custom_objects=None,
    fps=8,  
    max_new_tokens=1024
):

    model, processor = setup_model_and_processor()
    
    prompt = create_phase_analysis_prompt(operation_type, custom_objects)
    

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


if __name__ == "__main__":
    video_path = "./video/cloth_sweeping_cropped_removed.mp4"  
    
    print("=== 机器人操作阶段分析 ===")
    result = analyze_operation_phases(
        video_path=video_path,
        operation_type="general",  
        custom_objects=["tool", "towel"],  # 可添加自定义物体，不填也可以
        fps=10 # 自定义采样帧率，精细操作需要更高采样率
    )
    
    print("分析结果:")
    print("=" * 50)
    print(result)
    

    video_name = os.path.basename(video_path)
    save_phase_results(result, video_name)
