#!/usr/bin/env python3
"""
Main entry for dual-agent interactive framework (generator/verifier).
Supports 3D (Blender) and 2D (PPTX) modes.
Uses MCP stdio connections instead of HTTP servers.
Includes a new VLM-based testing mode that generates two codes per iteration.

VLM Testing Mode Usage:
    python blendergym.py --vlm-test-mode --mode blendergym --max-rounds 5
    python blendergym.py --mode vlm-test --max-rounds 5

This mode generates two codes per iteration, executes both, and uses a VLM to select
which generated image is closer to the target image.
"""
import argparse
import os
import sys
import shutil
import time
import json
import asyncio
import base64
import io
from pathlib import Path
from typing import Optional, Tuple
from contextlib import AsyncExitStack
from openai import OpenAI
from PIL import Image
from utils.clients import GeneratorAgentClient, VerifierAgentClient

# ========== VLM Selection Function ==========

def get_image_base64(image_path: str) -> str:
    """Convert image to base64 data URL for OpenAI API."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        else:
            mime_type = 'image/png'  # default
        return f"data:{mime_type};base64,{encoded_string}"

async def select_better_image_with_vlm(
    left_image_path: str,
    right_image_path: str,
    target_image_path: str,
    vision_model: str,
    api_key: str,
    api_base_url: Optional[str] = None
) -> str:
    """
    Use VLM to select which image (left or right) is closer to the target.
    
    Args:
        left_image_path: Path to the left image
        right_image_path: Path to the right image  
        target_image_path: Path to the target image
        vision_model: OpenAI vision model name
        api_key: OpenAI API key
        api_base_url: Optional custom API base URL
        
    Returns:
        "left" or "right" indicating which image is better
    """
    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if api_base_url:
        client_kwargs["base_url"] = api_base_url
    
    client = OpenAI(**client_kwargs)
    
    # Convert images to base64
    target_b64 = get_image_base64(target_image_path)
    left_b64 = get_image_base64(left_image_path)
    right_b64 = get_image_base64(right_image_path)
    
    # Create the prompt
    prompt = """You are comparing two generated images (left and right) to determine which one is closer to the target image. 

Please analyze:
1. Overall composition and layout similarity to the target
2. Object positioning and arrangement
3. Visual style and appearance
4. Color schemes and lighting
5. Any specific details that match or differ from the target

Respond with ONLY "left" or "right" (no other text) to indicate which generated image is closer to the target."""
    
    try:
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": target_b64}
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": left_b64}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": right_b64}
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        
        choice = response.choices[0].message.content.strip().lower()
        if choice in ["left", "right"]:
            return choice
        else:
            print(f"Warning: VLM returned unexpected response '{choice}', defaulting to 'left'")
            return "left"
            
    except Exception as e:
        print(f"Error in VLM selection: {e}")
        print("Defaulting to 'left' selection")
        return "left"

# ========== Main Dual-Agent Loop ==========

async def main():
    parser = argparse.ArgumentParser(description="Dual-agent interactive framework")
    parser.add_argument("--mode", choices=["blendergym", "autopresent", "blendergym-hard", "demo", "design2code", "vlm-test"], default="blendergym", help="Choose 3D (Blender), 2D (PPTX), Design2Code mode, or VLM testing mode")
    parser.add_argument("--vlm-test-mode", action="store_true", help="Enable VLM-based testing mode (generates two codes per iteration and uses VLM to select better one)")
    parser.add_argument("--vision-model", default="gpt-4o", help="OpenAI vision model")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max interaction rounds")
    parser.add_argument("--init-code-path", default="data/blendergym/blendshape1/start.py", help="Path to initial code file")
    parser.add_argument("--init-image-path", default="data/blendergym/blendshape1/renders/start", help="Path to initial images")
    parser.add_argument("--target-image-path", default="data/blendergym/blendshape1/renders/goal", help="Path to target images")
    parser.add_argument("--target-description", default=None, help="Target description for 2D mode")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--task-name", default="blendshape", help="Task name for hints extraction")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    
    # Agent server paths 
    parser.add_argument("--generator-script", default="agents/generator_mcp.py", help="Generator MCP script path")
    parser.add_argument("--verifier-script", default="agents/verifier_mcp.py", help="Verifier MCP script path")
    
    # Blender execution parameters (for generator)
    parser.add_argument("--blender-server-path", default="servers/generator/blender.py", help="Path to Blender MCP server script")
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default="data/blendergym/blendshape1/blender_file.blend", help="Blender template file")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--save-blender-file", action="store_true", help="Save blender file")
    parser.add_argument("--meshy_api_key", default=os.getenv("MESHY_API_KEY"), help="Meshy API key")
    parser.add_argument("--va_api_key", default=os.getenv("VA_API_KEY"), help="VA API key")
    
    # Slides execution parameters (for generator)
    parser.add_argument("--slides-server-path", default="servers/generator/slides.py", help="Path to Slides MCP server script")
    
    # Tool server paths (for verifier)
    parser.add_argument("--image-server-path", default="servers/verifier/image.py", help="Path to image processing MCP server script")
    parser.add_argument("--scene-server-path", default="servers/verifier/scene.py", help="Path to scene investigation MCP server script")
    
    # HTML execution parameters (for generator)
    parser.add_argument("--html-server-path", default="servers/generator/html.py", help="Path to HTML execution MCP server script")
    parser.add_argument("--browser-command", default="google-chrome", help="Browser command for HTML screenshots")
    
    args = parser.parse_args()

    # Prepare output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare target description
    if args.target_description:
        if os.path.exists(args.target_description):
            with open(args.target_description, 'r') as f:
                target_description = f.read().strip()
        else:
            target_description = args.target_description
    else:
        target_description = None
        
    if args.save_blender_file:
        save_blender_file = args.output_dir + "/blender_file.blend"
        if not os.path.exists(save_blender_file):
            # copy the blender file to the output directory
            shutil.copy(args.blender_file, save_blender_file)

    # Init agents
    generator = GeneratorAgentClient(args.generator_script)

    try:
        # Connect to agents
        await generator.connect()

        # Create generator session - pass all args as kwargs
        generator_params = {
            "mode": args.mode,
            "vision_model": args.vision_model,
            "api_key": args.api_key,
            "task_name": args.task_name,
            "max_rounds": args.max_rounds,
            "init_code_path": args.init_code_path,
            "init_image_path": args.init_image_path,
            "target_image_path": args.target_image_path,
            "target_description": target_description,
            "api_base_url": args.openai_base_url,
            "thought_save": args.output_dir + "/generator_thoughts.json",
            "gpu_devices": args.gpu_devices,
            # Blender executor parameters
            "blender_server_path": args.blender_server_path,
            "blender_command": args.blender_command,
            "blender_file": args.blender_file,
            "blender_script": args.blender_script,
            "render_save": args.output_dir + "/renders",
            "script_save": args.output_dir + "/scripts",
            "blender_save": args.output_dir + "/blender_file.blend" if args.save_blender_file else None,
            "meshy_api_key": args.meshy_api_key,
            "va_api_key": args.va_api_key,
            # Slides executor parameters
            "slides_server_path": args.slides_server_path,
            "output_dir": args.output_dir,
            # HTML executor parameters
            "html_server_path": args.html_server_path,
        }
        
        await generator.create_session(**generator_params)
        
        # Main loop
        for round_num in range(args.max_rounds):
            print(f"\n=== Round {round_num+1} ===")
            
            # VLM Testing Mode: Generate two codes and use VLM to select better one
            print("Step 1: Generating first code...")
            gen_result_1 = await generator.call(no_memory=True)
            if gen_result_1.get("status") == "max_rounds_reached":
                print("Max rounds reached. Stopping.")
                break
            if gen_result_1.get("status") == "error":
                print(f"Generator error: {gen_result_1['error']}")
                break
            
            print("Step 2: Generating second code...")
            gen_result_2 = await generator.call(no_memory=True)
            if gen_result_2.get("status") == "max_rounds_reached":
                print("Max rounds reached. Stopping.")
                break
            if gen_result_2.get("status") == "error":
                print(f"Generator error: {gen_result_2['error']}")
                break
            
            # Extract codes from results
            code_1 = gen_result_1.get("code") or gen_result_1.get("current_code") or gen_result_1.get("generated_code")
            code_2 = gen_result_2.get("code") or gen_result_2.get("current_code") or gen_result_2.get("generated_code")
            
            if not code_1 or not code_2:
                print("Failed to generate both codes")
                break
            
            print(f"Generated code 1 (truncated):\n{code_1[:200]}...")
            print(f"Generated code 2 (truncated):\n{code_2[:200]}...")
            
            # Execute both codes and get results
            result_1 = None
            result_2 = None
            
            if gen_result_1.get("execution_result") and gen_result_1["execution_result"].get("status") == "success":
                result_1 = gen_result_1["execution_result"].get("result", {})
                if result_1.get("status") == "success":
                    print("✅ First code execution successful")
                else:
                    print(f"❌ First code execution failed: {result_1.get('output', 'Unknown error')}")
                    result_1 = None
            
            if gen_result_2.get("execution_result") and gen_result_2["execution_result"].get("status") == "success":
                result_2 = gen_result_2["execution_result"].get("result", {})
                if result_2.get("status") == "success":
                    print("✅ Second code execution successful")
                else:
                    print(f"❌ Second code execution failed: {result_2.get('output', 'Unknown error')}")
                    result_2 = None
            
            if not result_1 and not result_2:
                print("Both code executions failed, skipping this round")
                continue
            elif not result_1:
                print("Only second code succeeded, using it")
                selected_code = code_2
                selected_result = result_2
            elif not result_2:
                print("Only first code succeeded, using it")
                selected_code = code_1
                selected_result = result_1
            else:
                # Both succeeded, use VLM to select better one
                print("Step 3: Using VLM to select better result...")
                
                # Get target image path
                target_image_path = os.path.join(args.target_image_path, 'render1.png')
                if not os.path.exists(target_image_path):
                    print(f"Target image not found: {target_image_path}")
                    selected_code = code_1  # default to first
                    selected_result = result_1
                else:
                    # Use VLM to select better image
                    left_image_path = os.path.join(result_1["output"], 'render1.png')
                    right_image_path = os.path.join(result_2["output"], 'render1.png')
                    
                    if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
                        print("Generated images not found, defaulting to first result")
                        selected_code = code_1
                        selected_result = result_1
                    else:
                        vlm_choice = await select_better_image_with_vlm(
                            left_image_path=left_image_path,
                            right_image_path=right_image_path,
                            target_image_path=target_image_path,
                            vision_model=args.vision_model,
                            api_key=args.api_key,
                            api_base_url=args.openai_base_url
                        )
                        
                        print(f"VLM selected: {vlm_choice}")
                        if vlm_choice == "left":
                            selected_code = code_1
                            selected_result = result_1
                        else:
                            selected_code = code_2
                            selected_result = result_2
            
                # Add render results to generator as feedback
                await generator.add_feedback(selected_result["output"])
            
            print("Step 5: Saving thought processes...")
            await generator.save_thought_process()
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            await generator.cleanup()
        except Exception as e:
            print(f"Warning: Generator cleanup failed: {e}")
    
    print("\n=== Dual-agent interaction finished ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Unexpected error: {e}")