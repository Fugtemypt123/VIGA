import os
import sys
import shutil

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import time
import argparse
import asyncio
import subprocess
from openai import OpenAI
from typing import List, Dict, Any, Optional
from runners.demo.init import initialize_3d_scene_from_image, load_scene_info, update_scene_info
from runners.demo.asset import AssetGenerator
from utils.blender.get_scene_info import get_scene_info, assets_data

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

class TestModeDemo:
    """Test Mode Demo Class - starts from existing Blender file and imports assets"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5", base_url: str = None):
        """
        Initialize test mode demo class
        
        Args:
            api_key: OpenAI API key (optional, defaults to environment variable)
            model: OpenAI model
            base_url: OpenAI base URL
        """
        self.openai_api_key = api_key
        self.model = model
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        kwargs = {'api_key': self.openai_api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.max_iterations = 20
        self.completed_objects = []
    
    def import_asset_to_scene(self, blender_file_path: str, asset_path: str, object_name: str, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        Import a downloaded asset into the Blender scene
        
        Args:
            blender_file_path: Path to the Blender file
            asset_path: Path to the asset file (e.g., .obj, .fbx, .blend)
            object_name: Name for the imported object
            location: Position "x,y,z"
            scale: Scale factor
            
        Returns:
            dict: Import result
        """      
        try:
            print(f"[TestModeDemo] Importing asset '{object_name}' from {asset_path}")
            
            # Create import script
            import_script = f"""
import bpy
import os

# Clear existing mesh objects (optional - comment out if you want to keep existing objects)
# bpy.ops.object.select_all(action='DESELECT')
# bpy.ops.object.select_by_type(type='MESH')
# bpy.ops.object.delete(use_global=False)

# Import the asset based on file extension
asset_path = r"{asset_path}"
if asset_path.endswith('.obj'):
    bpy.ops.import_scene.obj(filepath=asset_path)
elif asset_path.endswith('.fbx'):
    bpy.ops.import_scene.fbx(filepath=asset_path)
elif asset_path.endswith('.blend'):
    bpy.ops.wm.append(filepath=asset_path)
else:
    print(f"Unsupported file format: {asset_path}")
    exit(1)

# Get the imported object(s) and rename the first one
imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
if imported_objects:
    main_object = imported_objects[0]
    main_object.name = "{object_name}"
    
    # Set location and scale
    main_object.location = ({location.split(',')[0]}, {location.split(',')[1]}, {location.split(',')[2]})
    main_object.scale = ({scale}, {scale}, {scale})
    
    print(f"Successfully imported {{main_object.name}} at location {{main_object.location}}")
else:
    print("No mesh objects found in the imported asset")
    exit(1)

# Save the file
bpy.ops.wm.save_mainfile()
print("Blender file saved successfully")
"""
            
            # Write import script to temporary file
            script_path = f"temp_import_{object_name}_{int(time.time())}.py"
            with open(script_path, 'w') as f:
                f.write(import_script)
            
            # Execute import script
            cmd = [
                'utils/blender/infinigen/blender/blender',
                "--background", blender_file_path,
                "--python", script_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Clean up temporary script
            os.remove(script_path)
            
            print(f"✓ Asset '{object_name}' imported successfully")
            
            return {
                "status": "success",
                "message": f"Asset '{object_name}' imported successfully",
                "object_name": object_name,
                "asset_path": asset_path,
                "location": location,
                "scale": scale
            }
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to import asset: {e}")
            return {
                "status": "error",
                "error": f"Import failed: {e.stderr if e.stderr else str(e)}"
            }
        except Exception as e:
            logging.error(f"Failed to import asset: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_main_iteration(self, blender_file_path: str, task_name: str, target_image_path: str, output_dir: str, max_rounds: int = 5, object_name: str = None) -> Dict[str, Any]:
        """
        Run main.py iteration for scene adjustment
        
        Args:
            blender_file_path: Path to the Blender file
            task_name: Task name
            target_image_path: Target image path
            output_dir: Output directory
            max_rounds: Maximum number of iteration rounds
            
        Returns:
            dict: Iteration result
        """
        level = task_name.split("-")[0]
        id = task_name.split("-")[1]
        map_id_name = {
            '1': 'christmas1',
            '2': 'meeting2',
            '3': 'outdoor3',
        }
        try:
            print(f"[TestModeDemo] Starting main.py iteration for scene adjustment")
            task_description = 'The new object is x=' + object_name + '.\n\nThat means you should ONLY edit the code related to bpy.data.objects[' + object_name + '].\n\n' + get_scene_info(task_name, blender_file_path)
            # Prepare main.py command
            cmd = [
                "python", "main.py",
                "--mode", "blendergym-hard",
                "--vision-model", self.model,
                "--api-key", self.openai_api_key,
                "--max-rounds", str(max_rounds),
                "--blender-file", blender_file_path,
                "--target-image-path", target_image_path,
                "--target-description", task_description,
                "--output-dir", output_dir,
                "--task-name", task_name,
                "--init-code-path", output_dir + f"/scripts/{object_name}/start.py",
                "--init-image-path", output_dir + f"/renders/{object_name}",
                "--blender-server-path", "servers/generator/blender.py",
                "--blender-command", "utils/blender/infinigen/blender/blender",
                "--blender-file", blender_file_path,
                "--blender-script", f'data/blendergym_hard/{level}/{map_id_name[id]}/pipeline_render_script.py',
                "--save-blender-file",
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=False, timeout=3600)  # 1 hour timeout
                print(f"Task completed successfully: {task_name}")
            except subprocess.CalledProcessError as e:
                error_msg = f"Task failed: {task_name}, Error: {e}"
                print(error_msg)
            except subprocess.TimeoutExpired:
                error_msg = f"Task timed out: {task_name}"
                print(error_msg)
            except Exception as e:
                error_msg = f"Task failed with exception: {task_name}, Error: {e}"
                print(error_msg)
                
            raise NotImplementedError("Not implemented")
                
        except Exception as e:
            logging.error(f"Failed to run main.py iteration: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_test_mode(self, blender_file_path: str, asset_paths: List[str], task_name: str, target_image_path: str, output_dir: str = "output/demo/test_mode") -> Dict[str, Any]:
        """
        Run test mode: import assets and iterate with main.py
        
        Args:
            blender_file_path: Path to existing Blender file
            asset_paths: List of asset file paths to import
            task_name: Task name
            target_image_path: Target image path
            output_dir: Output directory
            
        Returns:
            dict: Test mode result
        """
        try:
            print("=" * 60)
            print("🧪 Starting Test Mode Demo")
            print("=" * 60)
            print(f"Blender file: {blender_file_path}")
            print(f"Asset paths: {asset_paths}")
            print(f"Task name: {task_name}")
            print(f"Target image: {target_image_path}")
            print(f"Output directory: {output_dir}")
            
            # Check if Blender file exists
            if not os.path.exists(blender_file_path):
                return {
                    "status": "error",
                    "error": f"Blender file not found: {blender_file_path}"
                }
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Import assets
            print("\n📦 Step 1: Importing assets...")
            import_results = []
            start_code_path = None
            
            for i, asset_path in enumerate(os.listdir(asset_paths)):
                if not os.path.exists(os.path.join(asset_paths, asset_path)):
                    print(f"⚠️ Asset file not found: {asset_path}")
                    continue
                
                # Generate object name from asset path
                object_name = asset_path
                object_dir = os.path.join(asset_paths, object_name)

                # Find .obj file in object_dir
                obj_file = [f for f in os.listdir(object_dir) if f.endswith('.obj')]
                if not obj_file:
                    print(f"⚠️ No .obj file found in {object_dir}")
                    continue
                obj_file = obj_file[0]
                
                # Import asset
                import_result = self.import_asset_to_scene(
                    blender_file_path=blender_file_path,
                    asset_path=os.path.join(object_dir, obj_file),
                    object_name=object_name,
                    location=f"{assets_data['location'][task_name][object_name]}",  # Spread objects along X axis
                    scale=assets_data['size'][task_name][object_name]
                )
                
                # copy current blender file to object directory
                os.makedirs(output_dir + f"/blender/{object_name}", exist_ok=True)
                shutil.copy(blender_file_path, output_dir + f"/blender/{object_name}/blender_file.blend")
                # delete duplicate blender file
                if os.path.exists(output_dir + f"/blender/{object_name}/blender_file.blend1"):
                    os.remove(output_dir + f"/blender/{object_name}/blender_file.blend1")
                
                import_results.append(import_result)
                self.completed_objects.append(object_name)
                
                if import_result.get("status") == "success":
                    print(f"✓ Imported: {object_name}")
                else:
                    print(f"❌ Failed to import: {object_name} - {import_result.get('error')}")
                    
                # Create a start code
                if not start_code_path:
                    start_code_path = output_dir + f"/scripts/{object_name}/start.py"
                    os.makedirs(output_dir + f"/scripts/{object_name}", exist_ok=True)
                    with open(start_code_path, "w") as f:
                        f.write(f"import bpy")
                else:
                    os.makedirs(output_dir + f"/scripts/{object_name}", exist_ok=True)
                    shutil.copy(start_code_path, output_dir + f"/scripts/{object_name}/start.py")
                    start_code_path = output_dir + f"/scripts/{object_name}/start.py"
                
                with open(start_code_path, "a") as f:
                    f.write(f"\n\nbpy.data.objects['{object_name}'].location = (0, 0, 0)\n\nbpy.data.objects['{object_name}'].rotation_euler = (0, 0, 0)\n\nbpy.data.objects['{object_name}'].scale = (1, 1, 1)\n\n")
                    
                # Run script to get the start image
                os.makedirs(output_dir + f"/renders/{object_name}", exist_ok=True)
                cmd = [
                    "utils/blender/infinigen/blender/blender",
                    "--background", blender_file_path,
                    "--python", os.path.dirname(asset_paths) + '/pipeline_render_script.py',
                    "--", os.path.dirname(asset_paths) + '/start.py',  output_dir + f"/renders/{object_name}"
                ]
                subprocess.run(cmd, check=True, capture_output=False, timeout=3600)

                # Step 2: Run main.py iteration
                print(f"\n🔄 Step 2: Running main.py iteration...")
                iteration_result = asyncio.run(self.run_main_iteration(
                    blender_file_path=blender_file_path,
                    task_name=task_name,
                    target_image_path=target_image_path,
                    output_dir=output_dir,
                    max_rounds=10,
                    object_name=object_name
                ))
            
                # Return final result
                final_result = {
                    "status": "success" if iteration_result.get("status") == "success" else "partial",
                    "message": "Test mode completed",
                    "blender_file_path": blender_file_path,
                    "imported_objects": self.completed_objects,
                    "import_results": import_results,
                    "iteration_result": iteration_result,
                    "output_dir": output_dir
                }
                
                print("\n" + "=" * 60)
                print("🎉 Test Mode Demo Completed!")
                print("=" * 60)
                print(f"Objects imported: {len(self.completed_objects)}")
                print(f"Final objects: {self.completed_objects}")
                print(f"Main.py status: {iteration_result.get('status')}")
                
                raise NotImplementedError("Not implemented")
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to run test mode: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

def run_demo(target_image_path: str, task_name: str, model: str = "gpt-5-2025-08-07", base_url: str = None, api_key: str = None, output_dir: str = "output/demo/") -> Dict[str, Any]:
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

def run_test_mode_demo(blender_file_path: str, asset_paths: List[str], task_name: str, target_image_path: str, model: str = "gpt-5-2025-08-07", base_url: str = None, api_key: str = None, output_dir: str = "output/demo/test_mode") -> Dict[str, Any]:
    """
    Run test mode demo: import assets and iterate with main.py
    
    Args:
        blender_file_path: Path to existing Blender file
        asset_paths: List of asset file paths to import
        task_name: Task name
        target_image_path: Target image path
        model: OpenAI model
        base_url: OpenAI base URL
        api_key: OpenAI API key
        output_dir: Output directory
        
    Returns:
        dict: Test mode result
    """
    try:
        # Check if Blender file exists
        if not os.path.exists(blender_file_path):
            return {
                "status": "error",
                "error": f"Blender file not found: {blender_file_path}"
            }
        
        # Check if target image exists
        if not os.path.exists(target_image_path):
            return {
                "status": "error",
                "error": f"Target image not found: {target_image_path}"
            }
        
        # Create test mode demo instance and run
        demo = TestModeDemo(api_key=api_key, model=model, base_url=base_url)
        result = demo.run_test_mode(blender_file_path, asset_paths, task_name, target_image_path, output_dir)
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to run test mode demo: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
        
def direct_run(task_name: str, model: str, api_key: str, output_dir: str, max_rounds: int, blender_file_path: str, target_image_path: str, task_description: str):
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir + "/" + task_name + "/" + time_stamp
    os.makedirs(output_dir, exist_ok=True)
    level = task_name.split('-')[0]
    id = task_name.split('-')[1]
    map_id_name = {
        '1': 'christmas1',
        '2': 'meeting2',
        '3': 'outdoor3',
    }
    task_description = {
        '1': "Place the objects in the Christmas scene in the correct positions, making sure that the objects are positioned correctly relative to each other and that none of them are stuck in the wall.",
        '2': "First, move the table and chair combination to the correct position, do not let the chair get stuck in the wall, then place the small object on top of the table. If you can't see the object, it means it is stuck under the table.",
        '3': "Place the objects in the outdoor scene in the correct positions, making sure that the objects are positioned correctly relative to each other and that none of them are stuck in the wall.",
        
    }
    cmd = [
        "python", "main.py",
        "--mode", "blendergym-hard",
        "--vision-model", model,
        "--api-key", api_key,
        "--max-rounds", str(max_rounds),
        "--blender-file", blender_file_path,
        "--target-image-path", target_image_path,
        "--target-description", task_description[id],
        "--output-dir", output_dir,
        "--task-name", task_name,
        "--init-code-path", os.path.dirname(blender_file_path) + "/start.py",
        "--init-image-path", os.path.dirname(blender_file_path) + "/renders/start",
        "--blender-server-path", "servers/generator/blender.py",
        "--blender-command", "utils/blender/infinigen/blender/blender",
        "--blender-script", f'data/blendergym_hard/{level}/{map_id_name[id]}/pipeline_render_script.py',
        "--save-blender-file",
    ]
    
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, timeout=3600)  # 1 hour timeout
        print(f"Task completed successfully: {task_name}")
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"Task failed: {task_name}, Error: {e}"
        print(error_msg)
        return error_msg
    except subprocess.TimeoutExpired:
        error_msg = f"Task timed out: {task_name}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Task failed with exception: {task_name}, Error: {e}"
        print(error_msg)
        return error_msg    

def test_demo():
    """
    Test demo functionality
    """
    parser = argparse.ArgumentParser(description="Test demo functionality")
    parser.add_argument("--direct-run", action="store_true", help="Direct run")
    parser.add_argument("--easy-mode", action="store_true", help="Enable test mode (start from existing Blender file)")
    # parser.add_argument("--blender-file", default='data/blendergym_hard/level4/meeting2/blender_file.blend', type=str, help="Path to existing Blender file (required for test mode)")
    # parser.add_argument("--asset-paths", default='data/blendergym_hard/level4/meeting2/assets', type=str, help="Paths to asset files to import (required for test mode)")
    parser.add_argument("--task-name", default="level4-2", type=str, help="Task name")
    # parser.add_argument("--target-image-path", default="data/blendergym_hard/level4/meeting2/renders/goal", type=str, help="Target image path")
    parser.add_argument("--max-rounds", default=20, type=int, help="Max rounds")
    parser.add_argument("--model", default="o4-mini", type=str, help="OpenAI model")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), type=str, help="OpenAI base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), type=str, help="OpenAI API key")
    parser.add_argument("--output-dir", default="output/demo", type=str, help="Output directory")
    args = parser.parse_args()
    
    map_id_name = {
        'level4-1': 'christmas1',
        'level4-2': 'meeting2',
        'level4-3': 'outdoor3',
    }
    
    args.blender_file = f'data/blendergym_hard/level4/{map_id_name[args.task_name]}/blender_file.blend'
    args.asset_paths = f'data/blendergym_hard/level4/{map_id_name[args.task_name]}/assets'
    args.target_image_path = f'data/blendergym_hard/level4/{map_id_name[args.task_name]}/renders/goal'
    
    print("🧪 Testing Scene Reconstruction Demo...")
    
    if args.direct_run:
        print("🔧 Running in DIRECT RUN MODE")
        result = direct_run(args.task_name, args.model, args.api_key, args.output_dir, args.max_rounds, args.blender_file, args.target_image_path, 'No task description')
        return {'status': 'success', 'result': result}
    
    # Check if test mode is enabled
    if args.easy_mode:
        print("🔧 Running in TEST MODE")
        
        # Validate test mode arguments
        if not args.blender_file:
            print("❌ Error: --blender-file is required for easy mode")
            return {"status": "error", "error": "--blender-file is required for test mode"}
        
        if not args.asset_paths:
            print("❌ Error: --asset-paths is required for easy mode")
            return {"status": "error", "error": "--asset-paths is required for easy mode"}
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Copy blender file to output directory
        shutil.copy(args.blender_file, args.output_dir)

        # Run easy mode demo
        try:
            result = run_test_mode_demo(
                blender_file_path=args.output_dir + "/blender_file.blend",
                asset_paths=args.asset_paths,
                task_name=args.task_name,
                target_image_path=args.target_image_path,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                output_dir=args.output_dir
            )
            print(f"\n📊 Easy Mode Result: {result.get('status', 'unknown')}")
            
            if result.get("status") in ["success", "partial"]:
                print(f"✓ Easy mode completed")
                print(f"  - Objects imported: {len(result.get('imported_objects', []))}")
                print(f"  - Main.py status: {result.get('iteration_result', {}).get('status', 'unknown')}")
            else:
                print(f"❌ Easy mode failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Easy mode error: {e}")
            return {"status": "error", "error": str(e)}
    
    else:
        print("🔧 Running in HARD MODE")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        
        # Run hard demo
        try:
            result = run_demo(target_image_path=args.target_image_path, task_name=args.task_name, model=args.model, base_url=args.base_url, api_key=args.api_key, output_dir=args.output_dir)
            print(f"\n📊 Hard Mode Result: {result.get('status', 'unknown')}")
            
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
