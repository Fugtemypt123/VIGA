# BlenderGym Data

## Render Command

```bash
# Replace paths with your local installation
$BLENDER_PATH --background $BLEND_FILE --python data/blendergym/pipeline_render_script.py -- $GOAL_SCRIPT output $OUTPUT_BLEND
```

Example with default paths:
```bash
utils/blender/infinigen/blender/blender --background data/blendergym/task/blender_file.blend --python data/blendergym/pipeline_render_script.py -- data/blendergym/task/goal.py test output.blend
```
