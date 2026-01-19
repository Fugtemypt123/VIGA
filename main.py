#!/usr/bin/env python3
"""Main entry for dual-agent interactive framework (generator/verifier).

Supports 3D (Blender) and 2D (PPTX) modes.
Uses MCP stdio connections instead of HTTP servers.
"""
import argparse
import asyncio
import logging
import os

from agents.generator import GeneratorAgent
from agents.verifier import VerifierAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the dual-agent interactive framework."""
    parser = argparse.ArgumentParser(description="Dual-agent interactive framework")
    parser.add_argument(
        "--mode",
        choices=["blendergym", "autopresent", "blenderstudio", "static_scene", "dynamic_scene"],
        required=True,
        help="Choose mode: blendergym, autopresent, blenderstudio, static_scene, or dynamic_scene",
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI vision model")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--api-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max interaction rounds")
    parser.add_argument("--memory-length", type=int, default=12, help="Memory length")
    parser.add_argument("--init-code-path", default=None, help="Path to initial code file")
    parser.add_argument("--init-image-path", default=None, help="Path to initial images")
    parser.add_argument("--target-image-path", default=None, help="Path to target images")
    parser.add_argument("--target-description", default=None, help="Target description")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--task-name", default=None, help="Task name for hints extraction")
    parser.add_argument("--assets-dir", default=None, help="Assets directory path for static_scene and dynamic_scene modes")
    parser.add_argument("--resource-dir", default=None, help="Task directory path for autopresent mode")
    parser.add_argument("--gpu-devices", default=os.getenv("CUDA_VISIBLE_DEVICES"), help="GPU devices for Blender")
    parser.add_argument("--clear-memory", action="store_true", help="Clear memory")
    parser.add_argument("--explicit-comp", action="store_true", help="Enable explicit completion")
    parser.add_argument("--no-tools", action="store_true", help="Use no tools mode")
    parser.add_argument("--init-setting", choices=["none", "minimal", "reasonable"], default="none", help="Setting for the static scene task")
    parser.add_argument("--prompt-setting", choices=["none", "procedural", "scene_graph", "get_asset", "init"], default="none", help="Setting for the prompt")
    parser.add_argument("--num-candidates", type=int, default=1, help="Number of candidates for the model")

    # Execution parameters
    parser.add_argument("--blender-command", default="utils/blender/infinigen/blender/blender", help="Blender command path")
    parser.add_argument("--blender-file", default=None, help="Blender template file")
    parser.add_argument("--blender-script", default="data/blendergym/pipeline_render_script.py", help="Blender execution script")
    parser.add_argument("--blender-save", default=None, help="Save blender file")
    parser.add_argument("--meshy_api_key", default=os.getenv("MESHY_API_KEY"), help="Meshy API key")
    parser.add_argument("--va_api_key", default=os.getenv("VA_API_KEY"), help="VA API key")
    parser.add_argument("--browser-command", default="google-chrome", help="Browser command for HTML screenshots")

    # Tool servers
    parser.add_argument("--generator-tools", default="tools/generator_base.py", help="Comma-separated list of generator tool server scripts")
    parser.add_argument("--verifier-tools", default="tools/verifier_base.py", help="Comma-separated list of verifier tool server scripts")

    args = parser.parse_args()
    args = vars(args)

    # Init agents
    logger.info("Initializing agents")
    verifier = VerifierAgent(args)
    await verifier.tool_client.connect_servers()
    generator = GeneratorAgent(args, verifier)
    logger.info("Agents initialized successfully")
    await generator.tool_client.connect_servers()

    try:
        # Main loop
        logger.info("Starting dual-agent interaction")
        await generator.run()
        logger.info("Dual-agent interaction finished")
    except Exception as e:
        logger.error("Error during execution: %s", e)
    finally:
        # Cleanup
        logger.info("Cleaning up")
        await verifier.cleanup()
        await generator.cleanup()
        logger.info("Cleanup finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error("Fatal error: %s", e)
