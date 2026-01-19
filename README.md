<div align="center">

## 👷 ⚠️ WARNING ⚠️ 👷

### 🚧 Work In Progress 🚧

**This project is currently under active development.**
**Unit tests are being implemented, and the build may be unstable.**
**Please check back later for the v0 release.**

</div>

<h1 align="center">VIGA: Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning</h1>

<p align="center">
    <a href="https://fugtemypt123.github.io/VIGA-website/"><img src="https://img.shields.io/badge/Page-Project-blue" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2601.11109"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv Paper"></a>
    <a href="https://huggingface.co/datasets/DietCoke4671/blenderbench"><img src="https://img.shields.io/badge/Benchmark-HuggingFace-yellow" alt="HuggingFace Benchmark"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</p>

<p align="center"><img src="docs/images/art_cropped.png" width="33%"><img src="docs/images/render.gif" width="33%"><img src="docs/images/dynamic.gif" width="33%"></p>

<br>

# About

VIGA is an analysis-by-synthesis code agent for programmatic visual reconstruction. It approaches vision-as-inverse-graphics through an iterative loop of generating, rendering, and verifying scenes against target images.

A single self-reflective agent alternates between two roles:

- **Generator** — Writes and executes scene programs using tools for planning, code execution, asset retrieval, and scene queries.

- **Verifier** — Examines rendered output from multiple viewpoints, identifies visual discrepancies, and provides feedback for the next iteration.

The agent maintains an evolving contextual memory with plans, code diffs, and render history. This write-run-compare-revise loop is self-correcting and requires no finetuning.

<p align="center">
    <img src="docs/images/trajectory.png" alt="VIGA Trajectory" width="100%">
</p>
