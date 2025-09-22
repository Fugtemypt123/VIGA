import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import orjson
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape


DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "output/blendergym_hard/gpt-4o/level1/camera1")).resolve()


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


# Mount static data directory for serving images and scripts read-only
app.mount("/data", StaticFiles(directory=str(DATA_ROOT), html=False), name="data")


def render_template(name: str, **context: Any) -> HTMLResponse:
    template = env.get_template(name)
    html = template.render(**context)
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    generator_json = read_json(DATA_ROOT / "generator_thoughts.json")
    scores = read_json(DATA_ROOT / "scores.json")
    ref_free_scores = read_json(DATA_ROOT / "reference_free_scores.json")
    rounds = discover_rounds(DATA_ROOT)
    return render_template(
        "index.html",
        request=request,
        data_root=str(DATA_ROOT),
        rounds=rounds,
        generator=generator_json,
        scores=scores,
        ref_free_scores=ref_free_scores,
    )


@app.get("/round/{round_id}", response_class=HTMLResponse)
def round_details(request: Request, round_id: int) -> HTMLResponse:
    rounds = discover_rounds(DATA_ROOT)
    if round_id not in rounds:
        raise HTTPException(status_code=404, detail="Round not found")
    info = get_round_info(DATA_ROOT, round_id)
    return render_template(
        "round.html",
        request=request,
        round_info=info,
        rounds=rounds,
    )



