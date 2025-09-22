import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import orjson
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape


# DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/demo/christmas1/20250921_070451")).resolve()

# Manually configured dataset roots (edit these placeholders)
DATA_ROOT_1 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/demo/christmas1/showcase1")).resolve()
DATA_ROOT_2 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/demo/christmas1/showcase2")).resolve()
DATA_ROOT_3 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/demo/level4-1/light_demo")).resolve()
DATA_ROOT_4 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/demo/level4-1/camera1")).resolve()
DATA_ROOT_5 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/demo/level4-1/camera2")).resolve()
DATA_ROOT_6 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/blendergym/place_demo_1/placement1")).resolve()
DATA_ROOT_7 = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/blendergym/place_demo_2/placement1")).resolve()
DATA_ROOT_8 = Path("/absolute/path/to/output/REPLACE_ME_8").resolve()
DATA_ROOT_9 = Path("/absolute/path/to/output/REPLACE_ME_9").resolve()

DATASETS: List[Path] = [
    DATA_ROOT_1,
    DATA_ROOT_2,
    DATA_ROOT_3,
    DATA_ROOT_4,
    DATA_ROOT_5,
    # DATA_ROOT_6,
    # DATA_ROOT_7,
    # DATA_ROOT_8,
    # DATA_ROOT_9,
]


def infer_dataset_name_from_path(path: Path) -> str:
    try:
        # Split by 'output' and take the trailing segment like 'demo/xxx/yyyy'
        parts = path.as_posix().split("/output/")
        return parts[-1] if parts and parts[-1] else path.name
    except Exception:
        return path.name


def read_json(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return orjson.loads(f.read())
    except FileNotFoundError:
        return None
    except Exception as e:
        raise e


def discover_rounds(root: Path) -> List[int]:
    renders_dir = root / "renders"
    if not renders_dir.exists():
        return []
    round_numbers: List[int] = []
    for entry in renders_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            round_numbers.append(int(entry.name))
    round_numbers.sort()
    return round_numbers


def get_round_info(root: Path, round_id: int) -> Dict[str, Any]:
    render_path = root / "renders" / str(round_id) / "render1.png"
    verifier_json = root / "verifier_thoughts" / f"{round_id}.json"
    script_path = root / "scripts" / f"{round_id}.py"
    info: Dict[str, Any] = {
        "round": round_id,
        "generator_image_rel": f"/data/renders/{round_id}/render1.png" if render_path.exists() else None,
        "verifier_json": read_json(verifier_json),
        "script_rel": f"/data/scripts/{round_id}.py" if script_path.exists() else None,
    }

    # investigator images under verifier_thoughts/investigator/... optional
    investigator_dir = root / "verifier_thoughts" / "investigator"
    investigator_images: List[str] = []
    if investigator_dir.exists():
        for sub in investigator_dir.rglob("*"):
            if sub.is_file() and sub.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"} and str(round_id) in sub.stem:
                try:
                    rel_path = sub.relative_to(root)
                    investigator_images.append(f"/data/{rel_path.as_posix()}")
                except Exception:
                    continue
    info["investigator_images"] = sorted(investigator_images)
    return info


app = FastAPI(title="AgenticVerifier Conversation Viewer")

templates_dir = Path(__file__).parent / "templates"
env = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    autoescape=select_autoescape(["html", "xml"]) 
)


# Mount static data directories /data1..9 when they exist
for i, p in enumerate(DATASETS, start=1):
    try:
        if p.exists():
            app.mount(f"/data{i}", StaticFiles(directory=str(p), html=False), name=f"data{i}")
    except Exception:
        # Skip mounting invalid paths
        pass


def render_template(name: str, **context: Any) -> HTMLResponse:
    template = env.get_template(name)
    html = template.render(**context)
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    # Determine selected dataset index (1..9)
    try:
        selected_idx = int(request.query_params.get("dataset") or "1")
    except ValueError:
        selected_idx = 1
    if selected_idx < 1 or selected_idx > len(DATASETS):
        selected_idx = 1

    selected_root = DATASETS[selected_idx - 1]

    # Build dataset list for sidebar; names can be adjusted by editing paths above
    def name_from_output(path: Path) -> str:
        s = path.as_posix()
        parts = s.split("/output/")
        return parts[-1] if len(parts) > 1 else s

    datasets: List[Dict[str, str]] = []
    for i, p in enumerate(DATASETS, start=1):
        label = name_from_output(p) if p else ""
        datasets.append({"index": str(i), "name": label})

    data_prefix = f"/data{selected_idx}"

    generator_json = read_json(selected_root / "generator_thoughts.json")
    rounds = discover_rounds(selected_root)
    return render_template(
        "index.html",
        request=request,
        data_root=str(selected_root),
        data_prefix=data_prefix,
        datasets=datasets,
        selected_dataset_index=str(selected_idx),
        rounds=rounds,
        generator=generator_json,
    )



