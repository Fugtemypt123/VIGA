# VIGA: Vision-as-Inverse-Graphics Agent

<p align="center">
  <a href="https://fugtemypt123.github.io/VIGA-website/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2601.11109"><img src="https://img.shields.io/badge/arXiv-VIGA-red" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/DietCoke4671/blenderbench"><img src="https://img.shields.io/badge/Benchmark-HF-yellow" alt="Hugging Face"></a>
</p>

## Introduction

VIGA is an **analysis-by-synthesis** code agent for programmatic visual reconstruction. It approaches vision-as-inverse-graphics through an iterative loop of generating, rendering, and verifying scenes against target images.

<p align="center">
  <img src="docs/images/method_new.png" alt="VIGA Method Overview" width="800">
</p>

A single self-reflective agent alternates between two roles:

- **Generator**: Writes and executes scene programs using tools for planning, code execution, asset retrieval, and scene queries.

- **Verifier**: Examines rendered output from multiple viewpoints, identifies visual discrepancies, and provides feedback for the next iteration.

The agent maintains an **evolving contextual memory** with plans, code diffs, and render history. This write → run → compare → revise loop is **self-correcting and requires no finetuning**, enabling the same protocol to run across different foundation VLMs.

## Supported Domains

VIGA naturally generalizes across 2D, 3D, and 4D visual tasks through its analysis-by-synthesis loop:

| Mode | Description | Output |
|------|-------------|--------|
| **[BlenderGym](https://github.com/richard-guyunqi/BlenderGym-Open)** | Single-step 3D graphics editing | Blender Python code |
| **[BlenderBench](https://huggingface.co/datasets/DietCoke4671/blenderbench)** | Multi-step 3D graphics editing (Level 1-3) | Blender Python code |
| **[SlideBench](https://github.com/para-lost/AutoPresent)** | 2D programmatic slide/document layout synthesis | PowerPoint (PPTX) |
| **Custom Static Scene** | Single-view 3D scene reconstruction from scratch | Blender scene (.blend) |
| **Custom Dynamic Scene** | 4D dynamic scene reconstruction with physics | Blender animation |


## Quick Setup

### 1. Clone & Init Submodules
```bash
git clone https://github.com/Fugtemypt123/VIGA-release.git && cd VIGA-release
git submodule update --init --recursive
```

### 2. Setup Environments
```bash
conda create -n agent python=3.10 -y && conda activate agent && pip install -r requirements/requirement_agent.txt
conda create -n blender python=3.11 -y && conda activate blender && pip install -r requirements/requirement_blender.txt
conda create -n sam python=3.10 -y && conda activate sam && pip install -r requirements/requirement_sam.txt
conda create -n sam3d python=3.11 -y && conda activate sam3d && pip install -r requirements/requirement_sam3d-objects.txt
```

### 3. Configure API Keys
```bash
cp utils/_api_keys.py.example utils/_api_keys.py
# Edit utils/_api_keys.py: add OPENAI_API_KEY and MESHY_API_KEY
```

### 4. Run
```bash
conda activate agent && python runners/dynamic_scene.py --task=artist --model=gpt-5
```

You can customize any data you like. Please refer to the format in `data/dynamic_scene/artist` (put your new data in `data/dynamic_scene/<your-data-name>`).

## Full Documentation

- [Architecture](docs/architecture.md) - System design and skill library
- [Conda Setup](requirements/readme.md) - Detailed Conda installation guide
- [Runners](runners/readme.md) - Batch execution and command-line options

## Citation

```bibtex
@misc{yin2026visionasinversegraphicsagentinterleavedmultimodal,
      title={Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning}, 
      author={Shaofeng Yin and Jiaxin Ge and Zora Zhiruo Wang and Xiuyu Li and Michael J. Black and Trevor Darrell and Angjoo Kanazawa and Haiwen Feng},
      year={2026},
      eprint={2601.11109},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.11109}, 
}
```
