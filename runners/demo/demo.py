import os
import json
import logging
import time
import argparse
from typing import List, Dict, Any, Optional
from .init import initialize_3d_scene_from_image, load_scene_info, update_scene_info
from .asset import AssetGenerator

class SceneReconstructionDemo:
    """Scene Reconstruction Demo Class"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize demo class
        
        Args:
            api_key: Meshy API key (optional, defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        
        self.current_scene = None
        self.asset_generator = None
        self.max_iterations = 10  # Maximum number of iterations
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
        try:
            # This should call VLM API to analyze current scene and target image
            # Now using simple simulation logic first
            
            print(f"[VLM Analysis] Analyzing scene vs target image...")
            print(f"  - Current scene objects: {len(current_scene_info.get('objects', []))}")
            print(f"  - Target image: {target_image_path}")
            
            # Simulate VLM analysis results
            # In actual implementation, you need to:
            # 1. Use VLM to analyze target image and identify objects in it
            # 2. Use VLM to analyze current scene and identify existing objects
            # 3. Compare the two to find missing objects
            
            # Simple simulation logic: assume common objects in target image
            target_objects = [
                "chair", "table", "lamp", "sofa", "bookshelf", 
                "coffee_table", "bed", "desk", "television", "plant"
            ]
            
            # Get existing objects in current scene
            current_objects = [obj.get("name", "").lower() for obj in current_scene_info.get("objects", [])]
            current_objects.extend(self.completed_objects)
            
            # Find missing objects
            missing_objects = []
            for obj in target_objects:
                if obj not in current_objects and obj not in self.completed_objects:
                    missing_objects.append(obj)
            
            # Limit to maximum 3 objects per iteration to avoid generating too many at once
            missing_objects = missing_objects[:3]
            
            print(f"[VLM Analysis] Found {len(missing_objects)} missing objects: {missing_objects}")
            
            return missing_objects
            
        except Exception as e:
            logging.error(f"Failed to analyze missing objects: {e}")
            return []
    
    def run_reconstruction_loop(self, target_image_path: str, output_dir: str = "output/demo/reconstruction") -> Dict[str, Any]:
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
            scene_init_result = initialize_3d_scene_from_image(target_image_path, output_dir)
            
            if scene_init_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": f"Failed to initialize scene: {scene_init_result.get('error')}"
                }
            
            self.current_scene = scene_init_result
            self.asset_generator = AssetGenerator(
                blender_path=scene_init_result["blender_file_path"],
                api_key=self.api_key
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
                        location=f"{len(self.completed_objects) * 2},0,0",  # Avoid overlap
                        scale=1.0
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

def run_demo(target_image_path: str, api_key: str = None, output_dir: str = "output/demo/") -> Dict[str, Any]:
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
        demo = SceneReconstructionDemo(api_key=api_key)
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
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), type=str, help="OpenAI API key")
    parser.add_argument("--output-dir", default="output/demo/christmas1", type=str, help="Output directory")
    args = parser.parse_args()
    
    print("🧪 Testing Scene Reconstruction Demo...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    # Run demo
    try:
        result = run_demo(target_image_path=args.target_image_path, api_key=args.api_key, output_dir=args.output_dir)
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
