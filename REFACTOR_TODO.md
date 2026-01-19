# VIGA Refactoring TODO

This document tracks all code quality issues that need to be fixed for open-source release.

## Progress Tracking

- [x] Phase 1: agents/ ✅ COMPLETED
- [x] Phase 2: tools/ ✅ COMPLETED
- [x] Phase 3: models/ ✅ COMPLETED
- [x] Phase 4: runners/ ✅ COMPLETED
- [x] Phase 5: visualization/ ✅ N/A (directory removed)
- [x] Phase 6: evaluators/ ✅ COMPLETED
- [x] Phase 7: utils/ ✅ COMPLETED
- [x] Phase 8: prompts/ ✅ COMPLETED
- [x] Phase 9: data/ ✅ COMPLETED
- [x] Phase 10: Open-source preparation ✅ COMPLETED

---

## Phase 1: agents/ ✅ COMPLETED

### Changes Made:
- **tool_client.py**: Added module docstring, fixed import order (removed unused `from re import T`, moved `contextlib` to stdlib section), added comprehensive docstrings and type hints to all classes and methods, translated Chinese comments to English
- **generator.py**: Added module docstring, fixed import order, added class and method docstrings, added type hints (`args: Dict[str, Any]`, return types), translated Chinese comments
- **verifier.py**: Added module docstring, fixed import order, added class and method docstrings, added type hints
- **prompt_builder.py**: Added module docstring, fixed import order, removed leading blank lines, added type hints (`Optional[Dict]`), added method docstrings

### 1.1 Type Annotations (FIXED)

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `generator.py` | 11 | `args` param lacks type | Add `args: Dict[str, Any]` |
| `generator.py` | 136 | `Dict` incomplete | Change to `Dict[str, Any]` |
| `generator.py` | 199 | Missing return type | Add `-> None` |
| `verifier.py` | 10 | `args` param lacks type | Add `args: Dict[str, Any]` |
| `verifier.py` | 122 | `Dict` incomplete | Change to `Dict[str, Any]` |
| `verifier.py` | 163 | Missing return type | Add `-> None` |
| `tool_client.py` | 49 | `args` should be `Optional[Dict]`, missing return type | Fix both |
| `tool_client.py` | 63 | Return type too vague | Be more specific |
| `prompt_builder.py` | 17 | `prompts` should be `Optional[Dict]` | Fix type hint |

### 1.2 Unused Imports

| File | Line | Issue |
|------|------|-------|
| `tool_client.py` | 5 | `from re import T` - UNUSED, never referenced |

### 1.3 Import Order

| File | Issue |
|------|-------|
| `tool_client.py` | `contextlib` (stdlib) comes after third-party imports |

### 1.4 Docstrings Missing

| File | Function | Issue |
|------|----------|-------|
| `generator.py` | `_update_memory()` | No docstring |
| `generator.py` | `_save_memory()` | No docstring |
| `verifier.py` | `_update_memory()` | No docstring |
| `verifier.py` | `_save_memory()` | No docstring |
| `tool_client.py` | `ServerHandle` class | Minimal docstring |
| `tool_client.py` | `call_tool()` | No docstring |
| `tool_client.py` | `ExternalToolClient` | Minimal docstring |

### 1.5 Chinese Comments

| File | Line | Comment |
|------|------|---------|
| `generator.py` | (check) | May have Chinese comments |

---

## Phase 2: tools/ ✅ COMPLETED

### Changes Made:

All files in tools/ have been refactored to follow the coding standards:

**MCP Server Files:**
- **exec_blender.py**: Added module docstring, fixed imports, added type hints to all functions and globals, added comprehensive docstrings
- **exec_html.py**: Added module docstring, fixed imports, added `Dict[str, object]` return types on MCP functions
- **exec_slides.py**: Added module docstring, fixed imports, added type hints and docstrings
- **investigator.py**: Added module docstring, fixed imports, added class/method docstrings, fixed MCP function signatures
- **meshy.py**: Added module docstring, fixed global type hints (`_image_cropper: Optional["ImageCropper"] = None`), added MCP function docstrings
- **rag.py**: Added module docstring, fixed global type hints, added MCP function docstrings
- **sam.py**: Added module docstring, split multiple imports to one per line, added type hints to globals, removed commented-out code
- **initialize_plan.py**: Added type hints to tool config globals (`Dict[str, object]`)

**Worker/Utility Files:**
- **sam_init.py**: Added module docstring, fixed imports, translated Chinese comments to English, added type hints
- **sam_worker.py**: Added module docstring, fixed imports, translated Chinese comments, added comprehensive type hints and docstrings
- **sam3_worker.py**: Added module docstring, fixed imports, removed trailing dead code
- **sam3d_worker.py**: Added module docstring, fixed imports, translated Chinese comments, added type hints
- **sam3d_worker_multi_obj.py**: Added module docstring, fixed imports, moved imports to top of file, removed trailing dead code

**Blender Scripts:**
- **fix_glb.py**: Added module docstring, translated Chinese comments to English, added type hints
- **import_glbs_to_blend.py**: Added module docstring, translated Chinese comments, added type hints
- **blender_script_generators.py**: Added module docstring, fixed bug (missing `import json` in generated camera focus script), added type hints
- **generator_base.py**: Added module docstring, fixed imports, added type hints
- **verifier_base.py**: Added module docstring, fixed imports, added type hints

**Knowledge Base Scripts:**
- **knowledge_base/deduplicate_knowledge.py**: Fixed import order, added proper type hints, fixed bare except clause
- **knowledge_base/meshy_builder.py**: Added module docstring, split multiple imports, added type hints and docstrings
- **knowledge_base/rag_builder.py**: Added module docstring, split multiple imports, added comprehensive type hints and docstrings

### Bug Fixes:
- **blender_script_generators.py**: Fixed missing `import json` in `generate_camera_focus_script()` - generated scripts used `json.dump` without importing json
- **exec_html.py**: Fixed test code calling `execute_and_evaluate(full_code=code)` when parameter was named `code`

### 2.1 Type Annotations (FIXED)

All type annotation issues have been resolved. MCP functions now use `Dict[str, object]` return types consistently.

### 2.2 Import Order (FIXED)

All files now follow the import order: stdlib → third-party → local. Multiple imports per line have been split.

### 2.3 God Files (Deferred)

Large files have been improved with better documentation but splitting is deferred to avoid breaking changes:

| File | Lines | Status |
|------|-------|--------|
| `investigator.py` | 603 | Improved with docstrings |
| `meshy.py` | 602 | Improved with docstrings |
| `rag.py` | 528 | Improved with docstrings |
| `exec_blender.py` | 485 | Improved with docstrings |

---

## Phase 3: models/ ✅ COMPLETED

### Changes Made:

All files in models/ have been refactored to follow the coding standards:

- **client_chat.py**: Added module docstring, added `-> None` return type to `main()`, added function docstring
- **client_vision.py**: Added module docstring, added `-> None` return type to `main()`, added function docstring
- **server.py**: Added module docstring, added `-> List[str]` return type to `build_command()`, added `-> None` return type to `main()`, added comprehensive docstrings, fixed dead code (removed unnecessary `isinstance(args, dict)` check on line 82)

### 3.1 Type Annotations (FIXED)

| File | Issue | Fix |
|------|-------|-----|
| `client_chat.py` | `main()` missing return type | Added `-> None` |
| `client_vision.py` | `main()` missing return type | Added `-> None` |
| `server.py` | `build_command()` missing return type | Added `-> List[str]` |
| `server.py` | `main()` missing return type | Added `-> None` |

### Bug Fixes:
- **server.py**: Removed dead code `args["served_model_name"] if isinstance(args, dict) else args.served_model_name` - argparse always returns Namespace, never dict

---

## Phase 4: runners/ ✅ COMPLETED

### Changes Made:

All files in runners/ have been refactored to follow the coding standards:

**Main Runner Files:**
- **dynamic_scene.py**: Added module docstring, added `Tuple` import, fixed type hints (`args: argparse.Namespace`), added `-> None` return type to `main()`, added comprehensive docstrings with Args sections
- **static_scene.py**: Added module docstring, fixed type hints, added docstrings with complete Args sections including `setting` parameter

**BlenderGym Runners (runners/blendergym/):**
- **ours.py**: Added module docstring, fixed type hints, added comprehensive docstrings with Args/Returns sections
- **baseline.py**: Added module docstring, fixed type hints, added complete Args documentation
- **alchemy.py**: Added module docstring, comprehensive docstrings for all functions
- **run_all_code.py**: Added module docstring (simple script)

**BlenderBench Runners (runners/blenderbench/):**
- **main.py**: Added module docstring, fixed type hints, added complete Args documentation
- **ours.py**: Added module docstring, fixed type hints, added docstrings
- **alchemy.py**: Already had comprehensive docstrings

**SlideBench Runners (runners/slidebench/):**
- **ours.py**: Added module docstring, fixed type hints, added complete Args documentation
- **baseline.py**: Added module docstring, added docstrings to functions
- **create_slide.py**: Added module docstring, added docstring and return type to nested `save_response` function
- **library/__init__.py**: Added module docstring, fixed relative imports (was using absolute imports causing import errors)
- **library/*.py**: All have proper module docstrings and function documentation

### Bug Fixes:
- **slidebench/library/__init__.py**: Fixed absolute imports (`from library.library_basic import *`) to relative imports (`from .library_basic import *`) - was causing `ModuleNotFoundError` at runtime

### 4.1 Type Annotations (FIXED)

All type annotation issues have been resolved. Functions now use proper type hints like `Tuple[str, bool, str]` and `args: argparse.Namespace`.

### 4.2 Import Order (FIXED)

All files now follow the import order: stdlib → third-party → local.

### 4.3 God Files (Deferred)

Large files have been improved with better documentation but splitting is deferred to avoid breaking changes:

| File | Lines | Status |
|------|-------|--------|
| `blendergym/alchemy.py` | 783 | Improved with docstrings |
| `blenderbench/alchemy.py` | 770 | Improved with docstrings |

---

## Phase 5: visualization/ ✅ N/A

**Status**: Directory was removed from the codebase (commit f35af1e). No action required.

---

## Phase 6: evaluators/ ✅ COMPLETED

### Changes Made:

**SlideBench Metrics (evaluators/slidebench/metrics/):**
- **__init__.py**: Added module docstring, fixed relative imports
- **position.py**: Added module docstring, type hints, moved test code to `if __name__` block
- **text.py**: Added module docstring, type hints, lazy model loading, moved test code
- **color.py**: Added module docstring, type hints to all functions
- **clip.py**: Added module docstring, type hints, lazy model loading

**SlideBench Evaluators (evaluators/slidebench/):**
- **evaluate.py**: Added module docstring, fixed path bugs (`autopresent` → `slidebench`)
- **evaluate_baseline.py**: Added module docstring, fixed path bugs
- **gather.py**: Added module docstring, fixed path bugs
- **gather_baseline.py**: Added module docstring, fixed path bugs
- **match.py**: Added module docstring
- **page_eval.py**: Has existing docstring
- **reference_free_eval.py**: Added module docstring

**BlenderGym Evaluators (evaluators/blendergym/):**
- All files already had proper module docstrings

**BlenderBench Evaluators (evaluators/blenderbench/):**
- **evaluate.py**: Added module docstring, fixed path bugs (`blenderstudio` → `blenderbench`)
- **evaluate_baseline.py**: Fixed path bugs
- **gather.py**: Added module docstring, fixed path bugs
- **gather_baseline.py**: Added module docstring, fixed path bugs
- **ref_based_eval.py**: Fixed path bugs
- **ref_free_eval.py**: Fixed path bugs

### Bug Fixes:
- **slidebench/**: Fixed all `evaluators/autopresent` → `evaluators/slidebench` path references
- **slidebench/**: Fixed all `data/autopresent` → `data/slidebench` path references
- **slidebench/**: Fixed all `output/autopresent` → `output/slidebench` path references
- **blenderbench/**: Fixed all `blenderstudio` → `blenderbench` path references (data, output, evaluators)

---

## Phase 7: utils/ ✅ COMPLETED

### Changes Made:

**Note**: Consider renaming `utils/` to `third_party/` - this folder contains external helper libraries (SlidesLib, library) rather than general project utilities.

**Core Utils Files:**
- **common.py**: Added module docstring, fixed import order (stdlib → third-party → local), moved `logging` import from inline to top, added return type hints to `get_model_response()`, `build_client()`, `get_model_info()`, `get_meshy_info()`, fixed "Rets:" to "Returns:" in docstrings
- **path.py**: Added module docstring explaining configuration purpose

**Library Submodule (utils/library/):**
- **__init__.py**: Added module docstring
- **get_docs.py**: Added module docstring
- **library.py**: Added module docstring, fixed import order, changed "Rets:" to "Returns:"
- **library_basic.py**: Added module docstring, fixed import order, changed "Rets:" to "Returns:"
- **library_image.py**: Added module docstring, changed "Rets:" to "Returns:"

**SlidesLib Submodule (utils/SlidesLib/):**
- **__init__.py**: Added module docstring, fixed import order (third-party → local), **removed duplicate RGBColor import**, fixed spacing in function signatures
- **image_gen.py**: Added module docstring, fixed import order, added class docstring and return type hints
- **llm.py**: Added module docstring, **removed unused `requests` import**, fixed import order, added return type hints, fixed spacing
- **plotting.py**: Added module docstring, fixed import order, **converted Sphinx-style docstrings to Google-style**, added return type hints, fixed incomplete `get_plot` method
- **ppt_gen.py**: Added module docstring, fixed import order
- **search.py**: Added module docstring, fixed import order, **removed duplicate `search_image_prev` method** (was defined twice identically), added return type hints, changed bare `except Exception:` to specific exceptions
- **vqa.py**: Added module docstring, **removed unused `requests` import**, moved inline imports to top of file, added return type hints

### Bug Fixes:
- **SlidesLib/__init__.py**: Removed duplicate `RGBColor` import (was imported twice)
- **SlidesLib/search.py**: Removed duplicate `search_image_prev` method definition (lines 86-113 were identical to 100-113)
- **SlidesLib/search.py**: Changed bare `except Exception:` to `except (requests.RequestException, IOError):` for proper error handling

---

## Phase 8: prompts/ ✅ COMPLETED

### Changes Made:

All files in prompts/ have been refactored to follow the coding standards:

**Core Files:**
- **prompt_manager.py**: Already had module docstring, type hints, and docstrings ✓
- **__init__.py**: Added module docstring

**Submodules (already compliant):**
- **blendergym/__init__.py, generator.py, verifier.py**: Already had module docstrings ✓
- **blenderbench/__init__.py, generator.py, verifier.py**: Already had module docstrings ✓
- **slidebench/__init__.py, generator.py, verifier.py**: Already had module docstrings ✓

**Fixed Files:**
- **static_scene/__init__.py**: Added module docstring
- **static_scene/generator.py**: Translated Chinese comments to English (removed internal development notes)
- **dynamic_scene/__init__.py**: Converted comment to module docstring
- **dynamic_scene/generator.py**: Converted comment to module docstring
- **dynamic_scene/verifier.py**: Converted comment to module docstring

### Bug Fixes:
- **runners/slidebench/**: Fixed all `autopresent` → `slidebench` path references (data paths, output paths)
- **runners/blenderbench/**: Fixed all `blenderstudio` → `blenderbench` path references (data paths, output paths)
- Note: Mode parameters kept as `autopresent` and `blenderstudio` for backward compatibility with main.py

---

## Phase 9: data/ ✅ COMPLETED

### Changes Made:

**data/blendergym/ (10 files):**
- **generator_script.py**: Added module docstring, removed unused imports
- **verifier_script.py**: Added module docstring, removed unused imports
- **pipeline_render_script.py**: Added module docstring, removed unused imports
- **all_render_script.py**: Added module docstring, **translated Chinese comments to English**, removed unused imports
- **eval_render_script.py**: Added module docstring, **translated Chinese comments to English**, removed unused imports
- **single_render_script.py**: Added module docstring, **translated Chinese comments to English**, removed unused imports
- **cp_blender_files.py**: Added module docstring
- **python_script.py**: Added module docstring, fixed import order, removed unused imports
- **rebuild_scene.py**: Added module docstring, **translated extensive Chinese comments to English** (both in script and generated rebuild template)
- **replace_import.py**: Added module docstring

**data/blenderbench/ (2 files):**
- **generator_script.py**: Added module docstring, removed unused imports
- **verifier_script.py**: Added module docstring, removed unused imports

**data/dynamic_scene/ (2 files):**
- **generator_script.py**: Added module docstring, **translated Chinese comments to English**
- **verifier_script.py**: Added module docstring, removed unused imports

**data/static_scene/ (3 files):**
- **generator_init_script.py**: Added module docstring, removed unused imports
- **generator_script.py**: Added module docstring, removed unused imports
- **verifier_script.py**: Added module docstring, removed unused imports

**data/slidebench/ (5 files):**
- **create_dataset.py**: Added module docstring, fixed import order
- **library.py**: Added module docstring, fixed import order, **changed "Rets:" to "Returns:" in all docstrings**
- **parse_media.py**: Added module docstring, fixed import order
- **reproduce_code.py**: Added module docstring, fixed import order
- **seed_instruction.py**: Added module docstring, fixed import order

### Summary:
- Added module docstrings to all 22 Python files
- Translated Chinese comments to English in 5 files
- Fixed import order (stdlib → third-party → local)
- Removed unused imports (random, json, platform)
- Fixed "Rets:" to "Returns:" in docstrings

---

## General Issues (All Files)

### Import Style
- Multiple imports on single line (violation of style guide)
- `sys.path.append` should be at top, before relative imports

### Exception Handling
- Replace generic `Exception` with specific exceptions
- Use `logging` module instead of `print` for errors

### Configuration
- Create `@dataclass` for typed configuration instead of `args: Dict`

---

## Phase 10: Open-source preparation ✅ COMPLETED

### Changes Made:

**Hardcoded Path Removal:**
- **utils/path.py**: Replaced all 14 hardcoded conda paths with `VIGA_CONDA_BASE` environment variable approach
- **requirements/requirement_sam.txt**: Changed absolute path to relative `./utils/sam`
- **requirements/requirement_sam3.txt**: Changed absolute path to relative `./utils/sam3`
- **requirements/requirement_sam3d-objects.txt**: Changed absolute path to relative `./utils/sam3d`
- **data/blendergym/README.md**: Updated command examples to use relative/template paths
- **data/blenderbench/README.md**: Updated command examples to use relative/template paths

**Import Order Fixes:**
- **evaluators/slidebench/reference_free_eval.py**: Reordered imports (stdlib → third-party → internal)
- **evaluators/blenderbench/ref_free_eval.py**: Reordered imports (stdlib → third-party → internal)
- **evaluators/slidebench/metrics/color.py**: Fixed import order and numpy compatibility patch

**Commented-Out Code Cleanup:**
- **prompts/blendergym/generator.py**: Removed old commented-out prompt template (23 lines)
- **prompts/static_scene/generator.py**: Removed old commented-out example loading code (30 lines)
- **data/slidebench/seed_instruction.py**: Removed old commented-out instruction
- **tools/verifier_base.py**: Removed commented parameter definitions
- **runners/blendergym/alchemy.py**: Removed commented exist_score check
- **runners/slidebench/create_slide.py**: Removed commented arguments
- **data/blendergym/generator_script.py**: Removed commented Blender code
- **data/blendergym/python_script.py**: Removed commented Blender code
- **data/static_scene/generator_init_script.py**: Removed commented render code

**Main Entry Point:**
- **main.py**: Removed Chinese comments, replaced print statements with logging module

**Configuration Templates:**
- Created **utils/_api_keys.py.example**: Template for API key configuration with proper instructions
- **utils/SlidesLib/ppt_gen.py**: Added complete type annotations to all 17 methods

### Summary:
- All paths are now relative or environment-variable based
- No hardcoded user paths remain in the codebase
- Import order follows the coding standard consistently
- All commented-out code has been cleaned up
- API key configuration template provided for easy setup

---

## Future Improvements (Optional)

These are optional improvements that can be done post-release:

### 1. Split Large Files (God Files)
Files exceeding 500 lines should be split by functionality:

| File | Lines | Suggested Split |
|------|-------|-----------------|
| `tools/investigator.py` | 603 | Split camera/render/viewpoint into separate modules |
| `tools/meshy.py` | 602 | Split API client/image processing/asset management |
| `runners/blendergym/alchemy.py` | 783 | Split evaluation/rendering/task management |
| `runners/blenderbench/alchemy.py` | 770 | Same as above |

### 2. Exception Handling
- Replace generic `except Exception:` with specific exception types
- Remove any remaining bare `except:` clauses

### 3. Configuration System
- Replace `args: Dict[str, Any]` with typed `@dataclass` configurations
```python
@dataclass
class AgentConfig:
    model: str
    max_rounds: int
    memory_length: int
    ...
```

### 4. Testing
- Add pytest unit tests for core modules (agents/, tools/)
- Add integration tests for the dual-agent loop

### 5. Documentation
- Generate API documentation (Sphinx/MkDocs)
- Add more usage examples to README.md
