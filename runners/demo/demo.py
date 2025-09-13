import os
import json
import logging
import time
import argparse
from openai import OpenAI
from typing import List, Dict, Any, Optional
from .init import initialize_3d_scene_from_image, load_scene_info, update_scene_info
from .asset import AssetGenerator

class SceneReconstructionDemo:
    """Scene Reconstruction Demo Class"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5", base_url: str = None):
        """
        Initialize demo class
        
        Args:
            api_key: Meshy API key (optional, defaults to environment variable)
        """
        self.openai_api_key = api_key
        self.meshy_api_key = os.getenv("MESHY_API_KEY")
        self.model = model
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        if not self.meshy_api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        
        kwargs = {'api_key': self.openai_api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.current_scene = None
        self.asset_generator = None
        self.max_iterations = 20  # Maximum number of iterations
        self.completed_objects = []  # List of completed objects
    
    def ask_vlm_for_missing_objects(self, current_scene_info: Dict[str, Any], target_image_path: str) -> List[str]:
        """
        Ask VLM which objects are missing in current scene compared to target scene
        
        Args:
            current_scene_info: Current scene information
            target_image_path: Target image path
            
        Returns:
            List[str]: List of missing object names
        """
        system_prompt = """You are a 3D scene expert. Now I will give you a picture and a list of objects I already have. Please find an object in the picture that does not appear in the list I have. You only need to output an object name, such as 'object: christmas tree'"""

        vlm_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": target_image_path}}, {"type": "text", "text": f"Objects I already have: {current_scene_info.get('objects', [])}"}]}
            ]
        )
        vlm_response = vlm_response.choices[0].message.content
        if 'object:' in vlm_response:
            vlm_response = vlm_response.split('object:')[1].strip()
        else:
            raise ValueError("VLM response is not a valid object name")
        return vlm_response
    
    def run_reconstruction_loop(self, target_image_path: str, output_dir: str = "output/demo") -> Dict[str, Any]:
        """
        Run scene reconstruction loop
        
        Args:
            target_image_path: Target image path
            output_dir: Output directory
            
        Returns:
            dict: Reconstruction result
        """
        try:
            print("=" * 60)
            print("🚀 Starting Scene Reconstruction Demo")
            print("=" * 60)
            print(f"Target image: {target_image_path}")
            print(f"Output directory: {output_dir}")
            
            # Step 1: Initialize 3D scene
            print("\n📋 Step 1: Initializing 3D scene...")
            scene_init_result = initialize_3d_scene_from_image(client=self.client, model=self.model, target_image_path=target_image_path, output_dir=output_dir)
            
            if scene_init_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": f"Failed to initialize scene: {scene_init_result.get('error')}"
                }
            
            self.current_scene = scene_init_result
            self.asset_generator = AssetGenerator(
                blender_path=scene_init_result["blender_file_path"],
                client=self.client,
                model=self.model
            )
            
            print(f"✓ Scene initialized: {scene_init_result['scene_name']}")
            
            # Step 2: Enter reconstruction loop
            print("\n🔄 Step 2: Starting reconstruction loop...")
            iteration = 0
            reconstruction_results = []
            
            while iteration < self.max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # Load current scene info
                scene_info = load_scene_info(scene_init_result["scene_info_path"])
                if not scene_info:
                    print("❌ Failed to load scene info")
                    break
                
                # Ask VLM for missing objects
                missing_objects = self.ask_vlm_for_missing_objects(scene_info, target_image_path)
                
                if not missing_objects:
                    print("✅ No missing objects found. Reconstruction complete!")
                    break
                
                print(f"🎯 Missing objects: {missing_objects}")
                
                # Generate assets for each missing object
                iteration_results = []
                for obj_name in missing_objects:
                    print(f"\n🔧 Generating assets for '{obj_name}'...")
                    
                    # Generate both types of assets (text and image)
                    asset_result = self.asset_generator.generate_both_assets(
                        object_name=obj_name,
                        image_path=target_image_path,  # Use target image as reference
                    )
                    
                    # Display result summary
                    summary = self.asset_generator.get_asset_summary(asset_result)
                    print(summary)
                    
                    iteration_results.append(asset_result)
                    
                    # Mark as completed
                    self.completed_objects.append(obj_name)
                
                reconstruction_results.append({
                    "iteration": iteration,
                    "missing_objects": missing_objects,
                    "results": iteration_results
                })
                
                # Update scene info
                scene_info["target_objects"].extend(missing_objects)
                update_scene_info(scene_init_result["scene_info_path"], scene_info)
                
                print(f"✓ Iteration {iteration} completed. Added {len(missing_objects)} objects.")
            
            # Step 3: Start scene editing (this part is left empty, waiting for subsequent implementation)
            print(f"\n🎨 Step 3: Starting scene editing (placeholder)...")
            editing_result = self.start_scene_editing(scene_init_result["blender_file_path"])
            
            # Return final result
            final_result = {
                "status": "success",
                "message": f"Scene reconstruction completed in {iteration} iterations",
                "scene_info": scene_init_result,
                "iterations": iteration,
                "completed_objects": self.completed_objects,
                "reconstruction_results": reconstruction_results,
                "editing_result": editing_result
            }
            
            print("\n" + "=" * 60)
            print("🎉 Scene Reconstruction Demo Completed!")
            print("=" * 60)
            print(f"Total iterations: {iteration}")
            print(f"Objects added: {len(self.completed_objects)}")
            print(f"Final objects: {self.completed_objects}")
            
            return final_result
            
        except Exception as e:
            logging.error(f"Failed to run reconstruction loop: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def start_scene_editing(self, blender_file_path: str) -> Dict[str, Any]:
        """
        Start scene editing (placeholder function, waiting for subsequent implementation)
        
        Args:
            blender_file_path: Blender file path
            
        Returns:
            dict: Editing result
        """
        try:
            print(f"[Scene Editing] Starting scene editing for: {blender_file_path}")
            print("[Scene Editing] This is a placeholder function - waiting for implementation")
            
            # This will call scene editing functionality in main.py in the future
            # Now return placeholder result first
            # TODO TODO TODO
            
            return {
                "status": "placeholder",
                "message": "Scene editing functionality not yet implemented",
                "blender_file_path": blender_file_path,
                "note": "This will be implemented in main.py"
            }
            
        except Exception as e:
            logging.error(f"Failed to start scene editing: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

def run_demo(target_image_path: str, model: str = "gpt-5-2025-08-07", base_url: str = None, api_key: str = None, output_dir: str = "output/demo/") -> Dict[str, Any]:
    """
    Run scene reconstruction demo
    
    Args:
        target_image_path: Target image path
        api_key: Meshy API key (optional)
        output_dir: Output directory
        
    Returns:
        dict: Demo result
    """
    try:
        # Check if input image exists
        if not os.path.exists(target_image_path):
            return {
                "status": "error",
                "error": f"Target image not found: {target_image_path}"
            }
        
        # Create demo instance and run
        demo = SceneReconstructionDemo(api_key=api_key, model=model, base_url=base_url)
        result = demo.run_reconstruction_loop(target_image_path, output_dir)
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to run demo: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def test_demo():
    """
    Test demo functionality
    """
    parser = argparse.ArgumentParser(description="Test demo functionality")
    parser.add_argument("--target-image-path", default="data/blendergym_hard/level4/christmas1/renders/goal/visprompt1.png", type=str, help="Target image path")
    parser.add_argument("--model", default="gpt-5", type=str, help="OpenAI model")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), type=str, help="OpenAI base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), type=str, help="OpenAI API key")
    parser.add_argument("--output-dir", default="output/demo/christmas1", type=str, help="Output directory")
    args = parser.parse_args()
    
    print("🧪 Testing Scene Reconstruction Demo...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    # Run demo
    try:
        result = run_demo(target_image_path=args.target_image_path, model=args.model, base_url=args.base_url, api_key=args.api_key, output_dir=args.output_dir)
        print(f"\n📊 Demo Result: {result.get('status', 'unknown')}")
        
        if result.get("status") == "success":
            print(f"✓ Demo completed successfully")
            print(f"  - Iterations: {result.get('iterations', 0)}")
            print(f"  - Objects added: {len(result.get('completed_objects', []))}")
        else:
            print(f"❌ Demo failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Demo test error: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Run test
    test_demo()
