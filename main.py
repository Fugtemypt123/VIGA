#!/usr/bin/env python3
"""
Main entry for dual-agent interactive framework (generator/verifier).
Supports 3D (Blender) and 2D (PPTX) modes.
Uses MCP stdio connections instead of HTTP servers.
"""
import argparse
import os
import shutil
import asyncio
from utils.clients import GeneratorAgentClient, VerifierAgentClient

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
    parser.add_argument("--assets-dir", default=None, help="Assets directory path for static_scene and dynamic_scene modes")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    
    # Agent server paths 
    parser.add_argument("--generator-script", default="agents/generator.py", help="Generator MCP script path")
    parser.add_argument("--verifier-script", default="agents/verifier.py", help="Verifier MCP script path")
    
    # Execution parameters
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default="data/blendergym/blendshape1/blender_file.blend", help="Blender template file")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--blender-save", default="output/test/static_scene/blender_file.blend", help="Save blender file")
    parser.add_argument("--meshy_api_key", default=os.getenv("MESHY_API_KEY"), help="Meshy API key")
    parser.add_argument("--va_api_key", default=os.getenv("VA_API_KEY"), help="VA API key")
    parser.add_argument("--browser-command", default="google-chrome", help="Browser command for HTML screenshots")
    
    # Generator/Verifier tool servers (comma-separated list of script paths)
    parser.add_argument("--generator-tools", default="tools/exec_blender.py,tools/meshy.py,tools/exec_html.py,tools/exec_slides.py,tools/rag.py", help="Comma-separated list of generator tool server scripts")
    
    # Verifier tool servers (comma-separated list of script paths)
    parser.add_argument("--verifier-tools", default="tools/init_verify.py,tools/investigator.py", help="Comma-separated list of verifier tool server scripts")
    
    args = parser.parse_args()

    # Prepare output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare target description
    if os.path.exists(args.target_description):
        with open(args.target_description, 'r') as f:
            args.target_description = f.read().strip()
    else:
        args.target_description = args.target_description 

    # Init agents
    generator = GeneratorAgentClient(args.generator_script)
    verifier = VerifierAgentClient(args.verifier_script)

    try:
        # Connect to agents
        await generator.connect()
        await verifier.connect()
        
        await generator.create_session(**args)
        await verifier.create_session(**args)
        
        # Main loop
        await generator.run()
            
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