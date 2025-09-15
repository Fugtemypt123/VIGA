origin_assets = {
    'level4-1': ['floor', 'left wall', 'right wall', 'CUADRO'] + ['bell', 'clock', 'fireplace', 'lounge', 'snowman', 'christmastree', 'giftboxes', 'decoration', 'goldenbells'],
    'level4-2': ['Door', 'window.006', 'dining-room_0-0.wall', 'dining-room_0-0.floor', 'dining-room_0-0.ceiling'] + ['table', 'chair.001', 'chair.002', 'chair.003', 'chair.004', 'chair.005', 'chair.006', 'chair.007', 'chair.008', 'openedbook', 'whitecup.001', 'whitecup.002', 'computer.001', 'computer.002', 'plant', 'blackcup.001', 'blackcup.002'],
}

add_assets = {
    'level4-1': ['clock', 'fireplace', 'lounge', 'snowman', 'christmastree', 'giftboxes', 'decoration', 'goldenbells'], 
    'level4-2': ['table', 'chair.001', 'chair.002', 'chair.003', 'chair.004', 'chair.005', 'chair.006', 'chair.007', 'chair.008', 'openedbook', 'whitecup.001', 'whitecup.002', 'computer.001', 'computer.002', 'plant', 'blackcup.001', 'blackcup.002'],
}


# # Golden bell on the wall
# bpy.data.objects['bell'].location = (-1.8349, 0.7151, 2.3103)
# bpy.data.objects['bell'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['bell'].scale = (0.4, 0.4, 0.4)

# # Clock on the wall
# bpy.data.objects['clock'].location = (-0.9259, 1.7568, 2.1043)
# bpy.data.objects['clock'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['clock'].scale = (0.3, 0.3, 0.3)

# # Fireplace at the corner
# bpy.data.objects['fireplace'].location = (-1.5455, 0.5751, 0.8951)
# bpy.data.objects['fireplace'].rotation_euler = (1.57, 0, 1.57)
# bpy.data.objects['fireplace'].scale = (0.9, 0.9, 0.9)

# # Lounge area in the middle
# bpy.data.objects['lounge area'].location = (0.0900, 0.8279, 0.6297)
# bpy.data.objects['lounge area'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['lounge area'].scale = (1.1, 1.1, 1.1)

# # Snowman on the table
# bpy.data.objects['snowman'].location = (0.1210, 0.1131, 0.7897)
# bpy.data.objects['snowman'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['snowman'].scale = (0.35, 0.35, 0.35)

# # Christmas tree in the corner
# bpy.data.objects['christmas_tree'].location = (-1.2528, -0.8087, 1.2191)
# bpy.data.objects['christmas_tree'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['christmas_tree'].scale = (1, 1, 1)

# # Box 1 near the christmas tree
# bpy.data.objects['box_inside'].location = (-0.4533, -0.8336, 0.4825)
# bpy.data.objects['box_inside'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['box_inside'].scale = (0.44, 0.44, 0.44)

# # Box 1 near the christmas tree
# bpy.data.objects['box_outside'].location = (-0.4533, -1.3499, 0.4825)
# bpy.data.objects['box_outside'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['box_outside'].scale = (0.44, 0.44, 0.44)

# # Tree decoration 1 on the wall
# bpy.data.objects['tree_decoration_inside'].location = (0.0869, 1.8155, 1.9509)
# bpy.data.objects['tree_decoration_inside'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['tree_decoration_inside'].scale = (0.3, 0.3, 0.3)

# # Tree decoration 2 on the wall
# bpy.data.objects['tree_decoration_outside'].location = (1.0199, 1.7982, 1.9414)
# bpy.data.objects['tree_decoration_outside'].rotation_euler = (1.57, 0, 0)
# bpy.data.objects['tree_decoration_outside'].scale = (0.3, 0.3, 0.3)

assets_data = {
    'size': {
        'level4-1': {
            'clock': (0.3, 0.3, 0.3),
            'fireplace': (0.9, 0.9, 0.9),
            'lounge': (1.1, 1.1, 1.1),
            'snowman': (0.35, 0.35, 0.35),
            'christmastree': (1, 1, 1),
            'giftboxes': (0.44, 0.44, 0.44),
            'decoration': (0.3, 0.3, 0.3),
            'goldenbells': (0.4, 0.4, 0.4),
        }
    },
    'rotation': {
        'level4-1': {
            'clock': (1.57, 0, 0),
            'fireplace': (1.57, 0, 1.57),
            'lounge': (1.57, 0, 0),
            'snowman': (1.57, 0, 0),
            'christmastree': (1.57, 0, 0),
            'giftboxes': (1.57, 0, 0),
            'decoration': (1.57, 0, 0),
            'goldenbells': (1.57, 0, 0),
        }
    },
}

def get_scene_info(task_name: str, blender_file_path: str) -> str:
    """
    Get scene information from Blender file by executing a script to list all objects.
    
    Args:
        blender_file_path: Path to the Blender file
        
    Returns:
        String containing scene information with object names
    """
    try:
        import bpy
        import mathutils
        
        # Clear existing scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Open the blender file (no saving; read-only introspection)
        bpy.ops.wm.open_mainfile(filepath=blender_file_path)
        
        # Get scene information
        scene_info = []
        
        print(bpy.context.scene.objects.keys())
        
        # List all objects in the scene
        scene_info.append("Scene Information:")
        for obj in bpy.context.scene.objects:
            obj_name = obj.name
            if task_name in origin_assets and obj_name not in origin_assets[task_name]:
                continue
            
            # Get object bounding box
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            bbox_min = mathutils.Vector((
                min(corner.x for corner in bbox_corners),
                min(corner.y for corner in bbox_corners),
                min(corner.z for corner in bbox_corners)
            ))
            bbox_max = mathutils.Vector((
                max(corner.x for corner in bbox_corners),
                max(corner.y for corner in bbox_corners),
                max(corner.z for corner in bbox_corners)
            ))
            bbox_size = bbox_max - bbox_min
            
            scene_info.append(f"- Name: {obj_name}; Location: {obj.location}; Rotation: {obj.rotation_euler}; Scale: {obj.scale}; BBox: min({bbox_min.x:.3f}, {bbox_min.y:.3f}, {bbox_min.z:.3f}), max({bbox_max.x:.3f}, {bbox_max.y:.3f}, {bbox_max.z:.3f})")
            # scene_info.append(f"- Name: {obj_name}; BBox: min({bbox_min.x:.3f}, {bbox_min.y:.3f}, {bbox_min.z:.3f}), max({bbox_max.x:.3f}, {bbox_max.y:.3f}, {bbox_max.z:.3f})")
            
        if len(scene_info) == 1:
            scene_info.append("All the information are provided in the code.")
        
        return "\n".join(scene_info)
        
    except ImportError:
        # If bpy is not available, return a placeholder message
        return "Scene information not available (Blender Python API not accessible)"
    except Exception as e:
        return f"Error getting scene information: {str(e)}"
    finally:
        # Ensure we do not leave a file open in Blender. Reset to factory settings silently.
        try:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        except Exception:
            # Suppress any cleanup errors to avoid shutdown issues
            pass