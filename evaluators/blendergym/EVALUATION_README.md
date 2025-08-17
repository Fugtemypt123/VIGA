# AgenticVerifier Evaluation Script

这个评估脚本用于评估 AgenticVerifier 在 BlenderGym 任务上的表现。

## 功能概述

评估脚本会：
1. 读取指定测试ID的输出结果
2. 将生成的图片与基准图片进行比较
3. 计算 CLIP 相似度和光度损失
4. 生成详细的评估报告

## 使用方法

### 基本用法

```bash
python evaluate.py <test_id>
```

例如：
```bash
python evaluate.py 20250815_150016
```

### 高级用法

```bash
python evaluate.py <test_id> --output_dir <custom_output_path>
```

例如：
```bash
python evaluate.py 20250815_150016 --output_dir /path/to/custom/evaluation
```

## 参数说明

- `test_id`: 测试ID，对应 `output/blendergym/` 下的目录名
- `--output_dir`: 可选，指定评估结果输出目录（默认为 `output/blendergym/{test_id}/evaluation`）

## 目录结构要求

### 输入目录结构
```
output/blendergym/{test_id}/
├── placement1/
│   ├── renders/
│   │   ├── 1/
│   │   │   ├── render1.png
│   │   │   └── render2.png (可选)
│   │   ├── 2/
│   │   │   ├── render1.png
│   │   │   └── render2.png (可选)
│   │   └── ...
│   └── ...
├── material5/
│   └── ...
└── ...
```

### 基准数据目录结构
```
data/blendergym/
├── placement1/
│   ├── renders/
│   │   └── goal/
│   │       ├── render1.png
│   │       └── render2.png (可选)
│   └── ...
├── material5/
│   └── ...
└── ...
```

## 输出结果

评估脚本会生成以下文件：

### 1. 整体评估结果 (`overall_scores.json`)
```json
{
    "placement": {
        "best_n_clip": 0.1234,
        "best_pl": 0.0567,
        "num_instances": 40
    },
    "material": {
        "best_n_clip": 0.0987,
        "best_pl": 0.0432,
        "num_instances": 40
    }
}
```

### 2. 详细评估结果 (`intermediate_scores.json`)
包含每个任务实例的详细分数信息。

### 3. 单个实例分数 (`{task_dir}/scores.json`)
每个任务目录下会生成一个 `scores.json` 文件，包含该实例所有轮次的详细分数。

## 评估指标

### 1. CLIP 相似度 (n_clip)
- 使用 CLIP 模型计算生成图片与目标图片的语义相似度
- 分数越低表示越相似
- 计算公式：`n_clip = 1 - clip_similarity(proposal_render, gt_render)`

### 2. 光度损失 (pl)
- 计算生成图片与目标图片的像素级差异
- 分数越低表示越相似
- 使用均方误差 (MSE) 计算

## 支持的任务类型

- `geometry`: 几何任务 (45个实例)
- `material`: 材质任务 (40个实例)
- `blendshape`: 形状混合任务 (75个实例)
- `placement`: 放置任务 (40个实例)
- `lighting`: 光照任务 (40个实例)

## 测试脚本

运行测试脚本来验证评估功能：

```bash
python test_evaluate.py
```

## 依赖项

确保安装了以下Python包：
```bash
pip install torch torchvision transformers pillow tqdm numpy
```

## 注意事项

1. 确保基准数据目录 `data/blendergym/` 存在且包含所有必要的目标图片
2. 评估过程可能需要较长时间，特别是当有很多任务实例时
3. CLIP 模型会在首次运行时下载，请确保网络连接正常
4. 如果某个任务实例缺少必要的图片文件，该实例会被跳过并显示警告信息

## 错误处理

脚本包含完善的错误处理机制：
- 自动跳过缺失的文件和目录
- 详细的警告和错误信息
- 优雅处理图片加载和处理错误
- 超时保护机制

## 示例输出

```
Found 240 task directories in output/blendergym/20250815_150016
Grouped tasks by type: ['placement', 'material', 'lighting']

Processing task type: placement
  Processing placement1 (instance 1)
    Best n_clip: 0.1234 (round 3)
    Best pl: 0.0567 (round 2)
  Processing placement2 (instance 2)
    Best n_clip: 0.0987 (round 1)
    Best pl: 0.0432 (round 4)
  ...

Task placement overall scores:
  Average best n_clip: 0.1111
  Average best pl: 0.0499
  Number of instances: 40

=== Evaluation Summary ===
Test ID: 20250815_150016
Output directory: output/blendergym/20250815_150016/evaluation
Results saved to: output/blendergym/20250815_150016/evaluation/overall_scores.json

PLACEMENT:
  Average best n_clip: 0.1111
  Average best pl: 0.0499
  Instances evaluated: 40
``` 