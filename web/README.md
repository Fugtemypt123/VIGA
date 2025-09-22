## AgenticVerifier Conversation Viewer

Run locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Configuration:
- Default data root: `../output/blendergym_hard/gpt-4o/level1/camera1`
- Override via env: `DATA_ROOT=/abs/path uvicorn app:app`

Open: http://localhost:8000



