# VIGA Refactoring TODO

This document tracks all code quality issues that need to be fixed for open-source release.

## Progress Tracking

- [x] Phase 1: agents/ ✅ COMPLETED
- [x] Phase 2: tools/ ✅ COMPLETED
- [x] Phase 3: models/ ✅ COMPLETED
- [x] Phase 4: runners/ ✅ COMPLETED
- [x] Phase 5: visualization/ ✅ N/A (directory removed)
- [x] Phase 6: evaluators/ ✅ COMPLETED
- [ ] Phase 7: utils/
- [ ] Phase 8: prompts/

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

## Phase 7: utils/

### 7.1 Exception Handling

| File | Line | Issue |
|------|------|-------|
| `common.py` | 26-27 | Generic `Exception` should be `RuntimeError` |

### 7.2 Hardcoded Values

| File | Line | Issue |
|------|------|-------|
| `common.py` | 19-28 | Hardcoded `max_retries = 1` and `time.sleep(10)` |

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
