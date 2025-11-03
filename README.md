# Robot Operation Phase Analysis Tool

## 概述

这是一个基于 Qwen3-VL 模型的机器人操作视频分析工具，能够自动识别机器人操作视频中的各个阶段，并提供详细的时间戳和描述。

## 功能特点

- 🎯 自动识别机器人操作的关键阶段
- ⏱️ 精确的时间戳标注（分:秒格式）
- 📝 详细的阶段描述和动作分析
- 💾 JSON格式输出，便于后续处理
- 🔧 可自定义物体词汇表和采样参数

## 安装依赖

```bash
pip install transformers torch av
```

## 使用方法

### 基本使用

```python
video_path = "your_video.mp4"
result = analyze_operation_phases(
    video_path=video_path,
    operation_type="general",
    custom_objects=["custom_tool", "special_part"],
    fps=10
)
```

### 参数说明

- `video_path`: 视频文件路径
- `operation_type`: 操作类型（当前固定为"general"）
- `custom_objects`: 自定义物体列表，可选
- `fps`: 视频采样帧率，默认为10

## Prompt 设计详解

### 核心功能

```python
prompt = f"""
You are a trainer explaining a robot's operation to a new technician. Watch the video and generate a step-by-step guide.
For each key step in the operation, identify the exact moment in the video where that step begins. Focus on moments where the robot's primary action changes."

Objects that may appear in the video include: {objects_text}
"""
```

### Prompt 设计策略

1. **角色设定**：将模型定位为"培训师"，要求以教学视角分析操作
2. **任务明确**：强调识别"关键步骤的开始时刻"和"主要动作变化点"
3. **上下文提供**：通过物体词汇表给模型提供领域知识
4. **输出约束**：严格指定JSON格式，确保结构化输出

### 输出格式要求

```json
{
    "phases": [
        {
            "phase_number": 1,
            "phase_name": "阶段名称",
            "start_time": "0:00",
            "end_time": "0:15", 
            "description": "详细的操作描述"
        }
    ],
    "summary": "整体操作摘要"
}
```

## 物体词汇表

工具内置了常见的机器人操作物体词汇：

```python
vocabularies = [
    "robot", "robotic arm", "end effector", "plate", "cup", "rack", "shelf",
    "table", "box", "container", "bowl", "screwdriver", "gripper", "clothes"
]
```

可通过 `custom_objects` 参数添加特定场景的物体。

## 输出示例

```json
{
    "phases": [
        {
            "phase_number": 1,
            "phase_name": "Initial Positioning",
            "start_time": "0:00",
            "end_time": "0:12",
            "description": "The robot arm moves to the starting position above the table, using its gripper to scan the workspace."
        }
    ],
    "summary": "The robot successfully completes a pick-and-place operation...",
    "metadata": {
        "video_file": "operation_video.mp4",
        "analysis_time": "2024-01-01 10:30:00"
    }
}
```

## 性能优化建议

1. **采样率选择**：
   - 一般操作：8-10 fps
   - 精细操作：12-15 fps  
   - 快速概览：4-6 fps

2. **内存管理**：模型加载后常驻内存，支持批量处理

3. **错误处理**：自动降级到文本格式保存，确保数据不丢失

## 注意事项

- 确保视频文件路径正确
- 首次运行需要下载模型权重（约15GB）
- 建议在GPU环境下运行以获得更好性能
- 输出文件保存在 `phase_results` 目录中