# Architecture

## Analysis-by-Synthesis Loop

VIGA operates through an iterative analysis-by-synthesis (AbS) loop where scene recovery (analysis) is achieved through generating candidate scenes (synthesis) and comparing them to target images. The agent is initialized with an empty context memory M₀ = ∅ and aims to produce a program p_T after T iterations:

```
┌─────────────────────────────────────────────────────────────┐
│                     Input (Target)                          │
│           (reference image / task description)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              SYNTHESIS: Generator Role                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • make_plan: Write high-level action plan            │   │
│  │ • execute_code: Run program to construct scene       │   │
│  │ • get_scene_info: Query object attributes            │   │
│  │ • get_better_assets: Retrieve/generate 3D assets     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Render Scene State                        │
│              st = exec(pt) → rendered image                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              ANALYSIS: Verifier Role                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • initialize_viewpoint: Compute canonical views      │   │
│  │ • set_camera / investigate: Explore from viewpoints  │   │
│  │ • Identify visual discrepancies vs target            │   │
│  │ • Generate structured feedback & suggestions         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Memory Update                              │
│     M_{t+1} = Tail_L(M_t || c_t)  (sliding window)          │
│   Retains most recent L iterations, reducing cost from      │
│   O(N²) to O(N·k) while keeping optimal reasoning range     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  writes → runs →      │
              │  compares → revises   │
              │    (Loop or STOP)     │
              └───────────────────────┘
```

## Project Structure

```
AgenticVerifier/
├── main.py                 # Main entry point for dual-agent loop
├── agents/                 # Core dual-agent implementation
│   ├── generator.py        # Generator agent logic
│   ├── verifier.py         # Verifier agent logic
│   ├── tool_client.py      # MCP tool client
│   └── prompt_builder.py   # System/user prompt construction
├── runners/                # Dataset runners for batch execution
│   ├── blendergym.py       # BlenderGym runner
│   ├── blenderstudio.py    # BlenderStudio runner
│   ├── static_scene.py     # Static scene runner
│   ├── dynamic_scene.py    # Dynamic scene runner
│   └── autopresent.py      # AutoPresent runner
├── tools/                  # Skill library (MCP tool servers)
│   ├── exec_blender.py     # execute_code: Blender execution & rendering
│   ├── exec_slides.py      # execute_code: PowerPoint generation
│   ├── exec_html.py        # execute_code: HTML/CSS rendering
│   ├── investigator.py     # Verification tools: initialize_viewpoint, set_camera, investigate
│   ├── meshy.py            # get_better_assets: 3D asset generation via Meshy API
│   ├── sam_init.py         # 3D scene reconstruction (SAM)
│   ├── initialize_plan.py  # make_plan: High-level action planning
│   ├── generator_base.py   # end_process for Generator
│   └── verifier_base.py    # end_process for Verifier
├── prompts/                # System and user prompts per mode
├── data/                   # Datasets for each mode
├── evaluators/             # Mode-specific evaluation scripts
├── utils/                  # Utility modules
├── requirements/           # Environment requirement files
└── output/                 # Output directory for runs
```

## Skill Library

VIGA uses a versatile skill library supporting both generation and verification (see Table 1 in paper):

| Tool | Role | Description |
|------|------|-------------|
| **Generation Tools** | |
| `make_plan` | Generator | Write an explicit high-level action plan before editing, helping the agent maintain long-horizon consistency instead of reacting myopically at each step |
| `execute_code` | Generator | Execute the current program to construct or update the scene, providing fresh renderings for the next verification step |
| `get_scene_info` | Generator | Query object attributes (e.g., position, rotation, scale) and summarize them as text, so the model can reason semantically about numeric parameters |
| `get_better_assets` | Generator | Retrieve or generate improved assets when existing ones are visually unsatisfactory, then insert them into the scene program |
| `end_process` | Generator | Signal that no further edits are needed and terminate the generation loop |
| **Verification Tools** | |
| `initialize_viewpoint` | Verifier | Given target objects, compute a joint bounding box and render canonical diagnostic views, exposing the scene from informative camera poses |
| `set_camera` | Verifier | Move the camera to a specified pose to inspect the scene from a particular viewpoint |
| `investigate` | Verifier | Adjust the camera using natural language commands (e.g., "rotate left", "zoom out"), turning low-level camera control into interpretable language actions |
| `set_keyframe` | Verifier | Navigate along the timeline in 4D scenes to inspect different keyframes and motion phases |
| `set_visibility` | Verifier | Toggle object visibility to reveal occluded content or isolate specific parts of the scene |
| `get_scene_info` | Verifier | Summarize the current scene structure and object states as text, which is then used to form feedback for the Generator |
| `end_process` | Verifier | Indicate that verification is complete and terminate the observation loop |

## Output Structure

Each task generates output capturing the evolving contextual memory (the iterative write→run→compare→revise trajectory):

```
output/<mode>/<timestamp>/<task_name>/
├── generator_thoughts/     # Generator synthesis logs (plans, code edits)
├── verifier_thoughts/      # Verifier analysis logs (discrepancies, suggestions)
├── renders/                # Rendered scene states per iteration
│   ├── 1/                  # s1 = exec(p1)
│   ├── 2/                  # s2 = exec(p2)
│   └── ...
├── codes/                  # Scene programs per iteration
├── scores.json             # Evaluation metrics
├── args.json               # Task configuration
└── blender_file.blend      # Final reconstructed scene (3D modes)
```

## Supported Models

VIGA supports multiple vision-language models (the same protocol works across heterogeneous foundation VLMs):

- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-4o-mini
- **Anthropic**: claude-sonnet-4, claude-opus-4.5
- **Google**: gemini-2.5-pro, gemini-2.0-flash
- **Qwen**: qwen-vl-max, qwen-vl-plus
