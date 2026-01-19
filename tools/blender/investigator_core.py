"""Core classes for 3D scene investigation.

Provides the Executor and Investigator3D classes for camera manipulation,
scene inspection, and viewpoint management in Blender scenes.
"""

import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from tools.blender.script_generators import (
    generate_scene_info_script,
    generate_render_script,
    generate_camera_focus_script,
    generate_camera_set_script,
    generate_visibility_script,
    generate_camera_move_script,
    generate_keyframe_script,
    generate_viewpoint_script
)


class Executor:
    """Lightweight executor for running Blender scripts.

    Handles script execution, rendering, and result collection for
    the investigator tool.

    Attributes:
        blender_command: Command to invoke Blender.
        blender_file: Path to the Blender scene file.
        blender_script: Path to the execution script.
        base: Base directory for outputs.
        script_path: Directory for saving scripts.
        render_path: Directory for rendered images.
        blender_save: Path to save modified Blender files.
        gpu_devices: CUDA device specification.
        count: Execution counter.
    """
    def __init__(
        self,
        blender_command: str,
        blender_file: str,
        blender_script: str,
        script_save: str,
        render_save: str,
        blender_save: Optional[str] = None,
        gpu_devices: Optional[str] = None
    ) -> None:
        """Initialize the executor.

        Args:
            blender_command: Command to invoke Blender.
            blender_file: Path to the Blender scene file.
            blender_script: Path to the execution script.
            script_save: Directory for saving scripts.
            render_save: Directory for rendered images.
            blender_save: Optional path to save modified Blender files.
            gpu_devices: Optional CUDA device specification.
        """
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.base = os.path.dirname(script_save)
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blender_save = blender_save
        self.gpu_devices = gpu_devices
        self.count = 0

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def next_run_dir(self) -> Path:
        """Create and return the next run directory.

        Returns:
            Path to the newly created run directory.
        """
        self.count += 1
        run_dir = self.render_path / f"{self.count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Clean old images if any
        for p in run_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        return run_dir

    def _execute_blender(self, code_file: Path, run_dir: Path) -> Dict[str, Any]:
        """Execute a Blender script and collect results.

        Args:
            code_file: Path to the Python script to execute.
            run_dir: Directory for output files.

        Returns:
            Dictionary with status and output (images or error text).
        """
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", str(code_file), str(run_dir)
        ]
        if self.blender_save:
            cmd.append(self.blender_save)

        env = os.environ.copy()
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        # Ban blender audio error
        env['AL_LIB_LOGLEVEL'] = '0'

        try:
            # Propagate render directory to scripts
            env["RENDER_DIR"] = str(run_dir)
            proc = subprocess.run(" ".join(cmd), shell=True, check=True, capture_output=True, text=True, env=env)
            imgs = sorted([str(p) for p in run_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            # If no image output
            if not os.path.exists(f"{self.base}/tmp/camera_info.json"):
                return {"status": "success", "output": {"text": [proc.stdout]}}
            # If image output
            with open(f"{self.base}/tmp/camera_info.json", "r") as f:
                camera_info = json.load(f)
                for camera in camera_info:
                    camera['location'] = [round(x, 2) for x in camera['location']]
                    camera['rotation'] = [round(x, 2) for x in camera['rotation']]
            return {"status": "success", "output": {"image": imgs, "text": ["Camera parameters: " + str(camera) for camera in camera_info]}}
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e.stderr}")
            return {"status": "error", "output": {"text": [e.stderr or e.stdout]}}

    def execute(self, full_code: str) -> Dict[str, Any]:
        """Execute Blender code and return results.

        Args:
            full_code: Complete Python code to execute in Blender.

        Returns:
            Dictionary with status and output (images or error text).
        """
        run_dir = self.next_run_dir()
        code_file = self.script_path / f"{self.count}.py"
        with open(code_file, "w") as f:
            f.write(full_code)
        result = self._execute_blender(code_file, run_dir)
        # Remove empty run directories
        if not os.listdir(run_dir):
            shutil.rmtree(run_dir)
            self.count -= 1
        return result


class Investigator3D:
    """3D scene investigator for camera manipulation and analysis.

    Provides methods for camera control, scene inspection, and viewpoint
    management in Blender scenes.

    Attributes:
        blender_file: Path to the Blender scene file.
        blender_command: Command to invoke Blender.
        base: Base directory for outputs.
        tmp_dir: Temporary directory for intermediate files.
        executor: Blender script executor.
        target: Current target object name for camera focus.
        radius: Camera orbit radius.
        theta: Camera azimuth angle.
        phi: Camera elevation angle.
        count: Operation counter.
        scene_info_cache: Cached scene information.
    """

    def __init__(
        self,
        save_dir: str,
        blender_path: str,
        blender_command: str,
        blender_script: str,
        gpu_devices: str
    ) -> None:
        """Initialize the 3D investigator.

        Args:
            save_dir: Directory for saving outputs.
            blender_path: Path to the Blender scene file.
            blender_command: Command to invoke Blender.
            blender_script: Path to the execution script.
            gpu_devices: CUDA device specification.
        """
        self.blender_file = blender_path
        self.blender_command = blender_command
        self.base = Path(save_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.base / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.executor = Executor(
            blender_command=blender_command,
            blender_file=blender_path,
            blender_script=blender_script,
            script_save=str(self.base / "scripts"),
            render_save=str(self.base / "renders"),
            blender_save=str(self.base / "current_scene.blend"),
            gpu_devices=gpu_devices
        )

        # Camera state variables
        self.target: Optional[str] = None
        self.radius: float = 5.0
        self.theta: float = 0.0
        self.phi: float = 0.0
        self.count: int = 0
        self.scene_info_cache: Optional[Dict[str, Any]] = None

    def _generate_scene_info_script(self) -> str:
        """Generate script to get scene information."""
        return generate_scene_info_script(f"{self.base}/tmp/scene_info.json")

    def _generate_render_script(self) -> str:
        """Generate script to render current scene once into RENDER_DIR/output.png."""
        return generate_render_script()

    def _generate_camera_focus_script(self, object_name: str) -> str:
        """Generate script to focus camera on object."""
        return generate_camera_focus_script(object_name, str(self.base))

    def _generate_camera_set_script(self, location: list, rotation_euler: list) -> str:
        """Generate script to set camera position and rotation."""
        return generate_camera_set_script(location, rotation_euler, str(self.base))

    def _generate_visibility_script(self, show_objects: list, hide_objects: list) -> str:
        """Generate script to set object visibility and render once."""
        return generate_visibility_script(show_objects, hide_objects, str(self.base))

    def _generate_camera_move_script(self, target_obj_name: str, radius: float, theta: float, phi: float) -> str:
        """Generate script to move camera around target object."""
        return generate_camera_move_script(target_obj_name, radius, theta, phi, str(self.base))

    def _generate_keyframe_script(self, frame_number: int) -> str:
        """Generate script to set frame number."""
        return generate_keyframe_script(frame_number, str(self.base))

    def _generate_viewpoint_script(self, object_names: list) -> str:
        """Generate script to initialize viewpoints around objects."""
        return generate_viewpoint_script(object_names, str(self.base))

    def _execute_script(self, script_code: str, description: str = "") -> dict:
        """Execute a blender script and return results."""
        try:
            result = self.executor.execute(full_code=script_code)

            # Update blender_background to the saved blend file
            if result.get("status") == "success":
                if self.executor.blender_save:
                    # Update the verifier base file
                    self.executor.blender_file = self.executor.blender_save

            return result
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return {"status": "error", "output": {"text": [str(e)]}}

    def _render(self) -> dict:
        """Render current scene and return image path and camera parameters."""
        render_script = self._generate_render_script()
        return self._execute_script(render_script, "Render current scene")

    def get_info(self) -> dict:
        """Get scene information by executing a script."""
        try:
            # Use cached info if available
            if self.scene_info_cache:
                return {"status": "success", "output": {"text": [str(self.scene_info_cache)]}}
            script = self._generate_scene_info_script()
            result = self._execute_script(script, "Extract scene information")
            if result.get("status") == "success":
                with open(f"{self.base}/tmp/scene_info.json", "r") as f:
                    scene_info = json.load(f)
                    self.scene_info_cache = scene_info
                    return {"status": "success", "output": {"text": [str(scene_info)]}}
            else:
                return {"status": "error", "output": {"text": ["Failed to extract scene information"]}}
        except Exception as e:
            return {"status": "error", "output": {"text": [str(e)]}}

    def focus_on_object(self, object_name: str) -> dict:
        """Focus camera on a specific object."""
        self.target = object_name  # Store object name instead of object reference
        # Generate and execute focus script
        focus_script = self._generate_camera_focus_script(object_name)
        result = self._execute_script(focus_script, f"Focus camera on object {object_name}")
        if os.path.exists(f"{self.base}/tmp/rotate_info.json"):
            with open(f"{self.base}/tmp/rotate_info.json", "r") as f:
                rotate_info = json.load(f)
                self.radius = rotate_info['radius']
                self.theta = rotate_info['theta']
                self.phi = rotate_info['phi']
        return result

    def zoom(self, direction: str) -> dict:
        """Zoom camera in or out."""
        if not self.target:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}
        if direction == 'in':
            self.radius = max(1, self.radius-3)
        elif direction == 'out':
            self.radius += 3
        return self._update_and_render()

    def move_camera(self, direction: str) -> dict:
        """Move camera around target object."""
        if not self.target:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}
        step = self.radius
        theta_step = step / (self.radius*math.cos(self.phi)) if math.cos(self.phi) != 0 else 0.1
        phi_step = step / self.radius
        if direction=='up':
            self.phi = min(math.pi/2-0.1, self.phi+phi_step)
        elif direction=='down':
            self.phi = max(-math.pi/2+0.1, self.phi-phi_step)
        elif direction=='left':
            self.theta -= theta_step
        elif direction=='right':
            self.theta += theta_step
        return self._update_and_render()

    def _update_and_render(self) -> dict:
        """Update camera position and render."""
        if not self.target:
            return self._render()
        # Generate script to move camera
        move_script = self._generate_camera_move_script(self.target, self.radius, self.theta, self.phi)
        return self._execute_script(move_script, f"Move camera around {self.target}")

    def set_camera(self, location: list, rotation_euler: list) -> dict:
        """Set camera position and rotation."""
        script = self._generate_camera_set_script(location, rotation_euler)
        return self._execute_script(script, f"Set camera to location {location} and rotation {rotation_euler}")

    def initialize_viewpoint(self, object_names: list) -> dict:
        """Initialize viewpoints around specified objects."""
        script = self._generate_viewpoint_script(object_names)
        return self._execute_script(script, f"Initialize viewpoints for objects: {object_names}")

    def set_keyframe(self, frame_number: int) -> dict:
        """Set scene to a specific frame."""
        script = self._generate_keyframe_script(frame_number)
        return self._execute_script(script, f"Set frame to {frame_number}")

    def set_visibility(self, show_objects: list, hide_objects: list) -> dict:
        """Set visibility of objects."""
        script = self._generate_visibility_script(show_objects, hide_objects)
        return self._execute_script(script, f"Set visibility: show {show_objects}, hide {hide_objects}")
