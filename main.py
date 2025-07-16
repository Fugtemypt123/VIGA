#!/usr/bin/env python3
"""
Main entry for dual-agent interactive framework (generator/verifier).
Supports 3D (Blender) and 2D (PPTX) modes.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional
import requests

api_key = os.getenv("OPENAI_API_KEY")

# ========== Agent Client Wrappers ==========

class GeneratorAgentClient:
    def __init__(self, url: str):
        self.url = url
        self.session_id = None
    def create_session(self, **kwargs):
        resp = requests.post(f"{self.url}/create_generation_session", json=kwargs)
        resp.raise_for_status()
        self.session_id = resp.json()["session_id"]
        return self.session_id
    def generate_code(self, feedback: Optional[str] = None):
        resp = requests.post(f"{self.url}/generate_code", json={"session_id": self.session_id, "feedback": feedback})
        resp.raise_for_status()
        return resp.json()
    def add_feedback(self, feedback: str):
        resp = requests.post(f"{self.url}/add_feedback", json={"session_id": self.session_id, "feedback": feedback})
        resp.raise_for_status()
    def save_thought_process(self):
        resp = requests.post(f"{self.url}/save_thought_process", json={"session_id": self.session_id})
        resp.raise_for_status()

class VerifierAgentClient:
    def __init__(self, url: str):
        self.url = url
        self.session_id = None
    def create_session(self, **kwargs):
        resp = requests.post(f"{self.url}/create_verification_session", json=kwargs)
        resp.raise_for_status()
        self.session_id = resp.json()["session_id"]
        return self.session_id
    def verify_scene(self, code: str, render_path: str, round_num: int):
        resp = requests.post(f"{self.url}/verify_scene", json={
            "session_id": self.session_id,
            "code": code,
            "render_path": render_path,
            "round_num": round_num
        })
        resp.raise_for_status()
        return resp.json()
    def save_thought_process(self):
        resp = requests.post(f"{self.url}/save_thought_process", json={"session_id": self.session_id})
        resp.raise_for_status()

# ========== Executor Wrappers ==========

class BlenderExecutorClient:
    def __init__(self, url: str):
        self.url = url
    def execute(self, code: str, round_num: int, **kwargs):
        resp = requests.post(f"{self.url}/exec_script", json={"code": code, "round": round_num, **kwargs})
        resp.raise_for_status()
        return resp.json()

class SlidesExecutorClient:
    def __init__(self, url: str):
        self.url = url
    def execute(self, code: str, round_num: int, code_save: str):
        resp = requests.post(f"{self.url}/exec_pptx", json={"code": code, "round": round_num, "code_save": code_save})
        resp.raise_for_status()
        return resp.json()

# ========== Main Dual-Agent Loop ==========

def main():
    parser = argparse.ArgumentParser(description="Dual-agent interactive framework")
    parser.add_argument("--mode", choices=["3d", "2d"], required=True, help="Choose 3D (Blender) or 2D (PPTX) mode")
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max interaction rounds")
    parser.add_argument("--init-code", required=True, help="Path to initial code file")
    parser.add_argument("--init-image-path", default=None, help="Path to initial images")
    parser.add_argument("--target-image-path", default=None, help="Path to target images")
    parser.add_argument("--generator-hints", default=None, help="Hints for generator agent")
    parser.add_argument("--verifier-hints", default=None, help="Hints for verifier agent")
    parser.add_argument("--thoughtprocess-save", default="thought_process.json", help="Path to save generator thought process")
    parser.add_argument("--verifier-thoughtprocess-save", default="verifier_thought_process.json", help="Path to save verifier thought process")
    parser.add_argument("--render-save", default="renders", help="Render save directory")
    parser.add_argument("--code-save", default="slides_code", help="Slides code save directory (2D mode)")
    parser.add_argument("--blender-save", default=None, help="Blender save path (3D mode)")
    parser.add_argument("--generator-url", default="http://localhost:8000", help="Generator agent server URL")
    parser.add_argument("--verifier-url", default="http://localhost:8004", help="Verifier agent server URL")
    parser.add_argument("--blender-url", default="http://localhost:8001", help="Blender executor server URL")
    parser.add_argument("--slides-url", default="http://localhost:8002", help="Slides executor server URL")
    args = parser.parse_args()

    # Read initial code
    with open(args.init_code, 'r') as f:
        init_code = f.read()

    # Prepare output dirs
    os.makedirs("output", exist_ok=True)
    os.makedirs(args.render_save, exist_ok=True)
    if args.mode == "2d":
        os.makedirs(args.code_save, exist_ok=True)

    # Init agents
    generator = GeneratorAgentClient(args.generator_url)
    verifier = VerifierAgentClient(args.verifier_url)

    # Create sessions
    generator.create_session(
        vision_model=args.vision_model,
        api_key=api_key,
        thoughtprocess_save=args.thoughtprocess_save,
        max_rounds=args.max_rounds,
        generator_hints=args.generator_hints,
        init_code=init_code,
        init_image_path=args.init_image_path,
        target_image_path=args.target_image_path,
        target_description=None
    )
    verifier.create_session(
        vision_model=args.vision_model,
        api_key=api_key,
        thoughtprocess_save=args.verifier_thoughtprocess_save,
        max_rounds=args.max_rounds,
        verifier_hints=args.verifier_hints,
        target_image_path=args.target_image_path,
        blender_save=args.blender_save
    )

    # Init executors
    if args.mode == "3d":
        executor = BlenderExecutorClient(args.blender_url)
    else:
        executor = SlidesExecutorClient(args.slides_url)

    # Main loop
    for round_num in range(args.max_rounds):
        print(f"\n=== Round {round_num+1} ===")
        # 1. Generator生成代码
        gen_result = generator.generate_code()
        if gen_result.get("status") == "max_rounds_reached":
            print("Max rounds reached. Stopping.")
            break
        if gen_result.get("status") == "error":
            print(f"Generator error: {gen_result['error']}")
            break
        code = gen_result["code"]
        print(f"Generated code (truncated):\n{code[:200]}...")
        # 2. 执行代码
        if args.mode == "3d":
            exec_result = executor.execute(
                code=code,
                round_num=round_num,
                blender_command="blender",
                blender_file="scene.blend",
                blender_script="render_script.py",
                script_save="scripts",
                render_save=args.render_save,
                blender_save=args.blender_save
            )
        else:
            exec_result = executor.execute(
                code=code,
                round_num=round_num,
                code_save=args.code_save
            )
        if exec_result.get("status") != "success":
            print(f"Execution failed: {exec_result.get('output')}")
            generator.add_feedback(f"Execution error: {exec_result.get('output')}")
            continue
        # 3. Verifier验证
        if args.mode == "3d":
            verify_result = verifier.verify_scene(
                code=code,
                render_path=args.render_save,
                round_num=round_num
            )
        else:
            # 2D模式可扩展为pptx图片路径
            verify_result = verifier.verify_scene(
                code=code,
                render_path=args.code_save,
                round_num=round_num
            )
        print(f"Verifier result: {verify_result.get('status')}")
        if verify_result.get("status") == "end":
            print("Verifier: OK! Task complete.")
            break
        elif verify_result.get("status") == "continue":
            feedback = verify_result["output"]
            print(f"Verifier feedback: {feedback}")
            generator.add_feedback(feedback)
        else:
            print(f"Verifier error: {verify_result.get('error')}")
            break
        # 4. 保存思考过程
        generator.save_thought_process()
        verifier.save_thought_process()
        time.sleep(1)
    print("\n=== Dual-agent interaction finished ===")

if __name__ == "__main__":
    main() 