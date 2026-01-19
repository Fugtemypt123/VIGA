<h1 align="center">VIGA: Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning</h1>

<p align="center">
    <a href="https://fugtemypt123.github.io/VIGA-website/"><img src="https://img.shields.io/badge/Page-Project-blue" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2601.11109"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv Paper"></a>
    <a href="https://huggingface.co/datasets/DietCoke4671/blenderbench"><img src="https://img.shields.io/badge/Benchmark-HuggingFace-yellow" alt="HuggingFace Benchmark"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</p>

<p align="center"><img src="docs/images/art_cropped.png" width="33%"><img src="docs/images/render.gif" width="33%"><img src="docs/images/dynamic.gif" width="33%"></p>

<p align="center">
    <a href="#about">About</a> •
    <a href="#supported-domains">Supported Domains</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#documentation">Documentation</a> •
    <a href="#citation">Citation</a>
</p>

<br>

# About

VIGA is an analysis-by-synthesis code agent for programmatic visual reconstruction. It approaches vision-as-inverse-graphics through an iterative loop of generating, rendering, and verifying scenes against target images.

A single self-reflective agent alternates between two roles:

- **Generator** — Writes and executes scene programs using tools for planning, code execution, asset retrieval, and scene queries.

- **Verifier** — Examines rendered output from multiple viewpoints, identifies visual discrepancies, and provides feedback for the next iteration.

The agent maintains an evolving contextual memory with plans, code diffs, and render history. This write-run-compare-revise loop is self-correcting and requires no finetuning.

<br>

# Supported Domains

| Mode | Description | Output |
|------|-------------|--------|
| [BlenderGym](https://github.com/richard-guyunqi/BlenderGym-Open) | Single-step 3D graphics editing | Blender Python |
| [BlenderBench](https://huggingface.co/datasets/DietCoke4671/blenderbench) | Multi-step 3D graphics editing (Level 1-3) | Blender Python |
| [SlideBench](https://github.com/para-lost/AutoPresent) | 2D slide/document layout synthesis | PowerPoint |
| Custom Static Scene | Single-view 3D reconstruction | Blender scene |
| Custom Dynamic Scene | 4D dynamic scene with physics | Blender animation |

<br>

# Quickstart

## 1. Installation: Setup the environment

### Prerequisites

You need [Conda](https://docs.conda.io/en/latest/miniconda.html) installed. For 3D modes, an NVIDIA GPU with CUDA support is recommended.

### Clone repository

```bash
git clone https://github.com/Fugtemypt123/VIGA-release.git && cd VIGA-release
git submodule update --init --recursive
```

### Create conda environments

VIGA requires separate environments for the agent and tools.

```bash
conda create -n agent python=3.10 -y && conda activate agent
pip install -r requirements/requirement_agent.txt

conda create -n blender python=3.11 -y && conda activate blender
pip install -r requirements/requirement_blender.txt

conda create -n sam python=3.10 -y && conda activate sam
pip install -r requirements/requirement_sam.txt

conda create -n sam3d python=3.11 -y && conda activate sam3d
pip install -r requirements/requirement_sam3d-objects.txt
```

See [Requirements](requirements/README.md) for additional options.

### Configure API keys

```bash
cp utils/_api_keys.py.example utils/_api_keys.py
```

Edit `utils/_api_keys.py` and add your `OPENAI_API_KEY` and `MESHY_API_KEY`.

## 2. Usage: Run the agent

```bash
conda activate agent
python runners/dynamic_scene.py --task=artist --model=gpt-5
```

Custom data: place in `data/dynamic_scene/<your-data-name>` following the format in `data/dynamic_scene/artist`.

<br>

# Documentation

| Doc | Description |
|-----|-------------|
| [Architecture](docs/architecture.md) | System design and agent tools |
| [Requirements](requirements/README.md) | Conda environment setup |
| [Runners](runners/README.md) | Batch execution options |

<br>

# Citation

You can find a paper writeup of the framework on [arXiv](https://arxiv.org/abs/2601.11109).

If you find this project useful for your research, please consider citing:

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
