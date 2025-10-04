#!/usr/bin/env python3
"""
Main entry for dual-agent interactive framework (generator/verifier).
Supports 3D (Blender) and 2D (PPTX) modes.
Uses MCP stdio connections instead of HTTP servers.
"""
import argparse
import os
import sys
import shutil
import time
import json
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import AsyncExitStack
from utils.clients import GeneratorAgentClient, VerifierAgentClient
from agents.config_manager import ConfigManager

# ========== Main Dual-Agent Loop ==========

async def main():
    parser = argparse.ArgumentParser(description="Dual-agent interactive framework")
    parser.add_argument("--mode", choices=["blendergym", "autopresent", "blendergym-hard", "demo", "design2code", "static_scene", "dynamic_scene"], default="blendergym", help="Choose 3D (Blender), 2D (PPTX), Design2Code, Static Scene, or Dynamic Scene mode")
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
    verifier = VerifierAgentClient(args.verifier_script)

    try:
        # Connect to agents
        await generator.connect()
        await verifier.connect()

        # Create configuration from args
        config_dict = {
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
            "output_dir": args.output_dir,
            # Server paths
            "blender_server_path": args.blender_server_path,
            "slides_server_path": args.slides_server_path,
            "html_server_path": args.html_server_path,
            "image_server_path": args.image_server_path,
            "scene_server_path": args.scene_server_path,
            # Blender-specific parameters
            "blender_command": args.blender_command,
            "blender_file": args.blender_file,
            "blender_script": args.blender_script,
            "save_blender_file": args.save_blender_file,
            "meshy_api_key": args.meshy_api_key,
            "va_api_key": args.va_api_key,
            "gpu_devices": args.gpu_devices,
        }
        
        # Create config manager and get setup configurations
        config_manager = ConfigManager(config_dict)
        
        # Get generator and verifier setup configurations
        generator_params = config_manager.get_executor_setup_config()
        verifier_params = config_manager.get_verifier_setup_config()
        
        await generator.create_session(**generator_params)
        await verifier.create_session(**verifier_params)

        # Main loop
        for round_num in range(args.max_rounds):
            print(f"\n=== Round {round_num+1} ===")
            
            print("Step 1: Generator generating code...")
            gen_result = await generator.call(no_memory=False)
            if gen_result.get("status") == "max_rounds_reached":
                print("Max rounds reached. Stopping.")
                break
            if gen_result.get("status") == "error":
                print(f"Generator error: {gen_result['error']}")
                break
            
            print("gen_result: ", gen_result)
            
            # Check if automatic execution happened
            if gen_result.get("execution_result"):
                exec_result = gen_result["execution_result"]
                if exec_result.get("status") == "success":
                    print("✅ Automatic execution completed!")
                    result = exec_result.get("result", {})
                    if result.get("status") == "success":
                        print("   - Generation successful")
                    else:
                        print(f"   - Generation failed: {result.get('output', 'Unknown error')}")
                        await generator.add_feedback(f"Execution error: {result.get('output')}")
                        continue
                else:
                    print(f"   - Execution failed: {exec_result.get('error', 'Unknown error')}")
                    await generator.add_feedback(f"Execution error: {exec_result.get('error')}")
                    continue
            else:
                # No execution result - this should not happen with new tool calling enforcement
                await generator.add_feedback("No execution result available. Please ensure you're calling the execute_and_evaluate tool.")
                continue
            
            # Add render results to generator as feedback
            await generator.add_feedback(result["output"])
            
            # Check if verifier should be called based on generator's flag
            call_verifier = gen_result.get("call_verifier", False)
            
            if call_verifier and gen_result.get("code"):
                print("Step 2: Verifier analyzing scene...")
                verify_result = await verifier.call(
                    code=gen_result["code"],
                    render_path=result["output"],
                    round_num=round_num,
                )
                
                print(f"Verifier result: {verify_result.get('status')}")
                if verify_result.get("status") == "end":
                    print("Verifier: OK! Task complete.")
                    break
                elif verify_result.get("status") == "continue":
                    feedback = verify_result["output"]
                    print(f"Verifier feedback: {feedback}")
                    await generator.add_feedback(feedback)
                else:
                    print(f"Verifier error: {verify_result.get('error')}")
                    break
            else:
                # Verifier not called or no code - add generic feedback
                if not call_verifier:
                    await generator.add_feedback("Code executed successfully. Continue with your modifications.")
                else:
                    await generator.add_feedback("No code execution, the state is the same as before.")
            
            print("Step 3: Saving thought processes...")
            await generator.save_thought_process()
            await verifier.save_thought_process()
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
        
        try:
            await verifier.cleanup()
        except Exception as e:
            print(f"Warning: Verifier cleanup failed: {e}")
    
    print("\n=== Dual-agent interaction finished ===")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Unexpected error: {e}")