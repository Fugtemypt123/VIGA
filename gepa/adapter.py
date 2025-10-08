from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar


# -----------------------------
# GEPA Engine Protocol types
# -----------------------------

DataInstT = TypeVar("DataInstT")
TrajectoryT = TypeVar("TrajectoryT")
RolloutOutputT = TypeVar("RolloutOutputT")


class EvaluationBatch(Generic[TrajectoryT, RolloutOutputT]):
    def __init__(
        self,
        outputs: List[RolloutOutputT],
        scores: List[float],
        trajectories: Optional[List[TrajectoryT]] = None,
    ) -> None:
        self.outputs = outputs
        self.scores = scores
        self.trajectories = trajectories


class GEPAAdapter(Protocol[DataInstT, TrajectoryT, RolloutOutputT]):
    def evaluate(
        self,
        batch: List[DataInstT],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TrajectoryT, RolloutOutputT]:
        ...

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[TrajectoryT, RolloutOutputT],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        ...


# -------------------------------------
# Adapter for AgenticVerifier codebase
# -------------------------------------


@dataclass
class AgenticDataInst:
    """Minimal task configuration for a single evaluation example.

    Fields are intentionally generic and JSON-serializable to be logged and
    rehydrated across runs.
    """

    mode: str
    target_image_path: Optional[str] = None
    output_dir: Optional[str] = None
    generator_tools: Optional[List[str]] = None
    verifier_tools: Optional[List[str]] = None
    task_name: Optional[str] = None
    # Arbitrary config passthrough to agents/config_manager
    extra_config: Optional[Dict[str, Any]] = None


# Types used for trajectories and rollout outputs
AgenticTrajectory = Dict[str, Any]
AgenticRollout = Dict[str, Any]


RunFn = Callable[[List[AgenticDataInst], Dict[str, str], bool], List[Tuple[AgenticRollout, float, Optional[AgenticTrajectory]]]]


class AgenticVerifierGEPAAdapter(
    Generic[AgenticDataInst, AgenticTrajectory, AgenticRollout]
):
    """GEPA adapter for AgenticVerifier.

    This adapter is deliberately lightweight and pluggable:
    - Provide a `run_fn` that executes your generator+verifier loop for a batch
      of data instances under a given `candidate` (prompt components), and returns
      a list of (output, score, trajectory) tuples.
    - If `run_fn` is not provided, a no-op runner is used that assigns zero
      scores, allowing GEPA to run end-to-end while you wire in execution later.

    Candidate contract (recommended keys):
    - "static_scene.generator.system" (str)
    - "static_scene.verifier.system" (str)
    - "dynamic_scene.generator.system" (str)
    - "dynamic_scene.verifier.system" (str)
    You can adopt any component keying scheme that your run function understands.
    """

    def __init__(self, run_fn: Optional[RunFn] = None, rng_seed: int = 7) -> None:
        self._run_fn = run_fn or self._default_noop_run
        self._rng = random.Random(rng_seed)

    # -----------------------------
    # 1) Program construction & eval
    # -----------------------------
    def evaluate(
        self,
        batch: List[AgenticDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[AgenticTrajectory, AgenticRollout]:
        try:
            results = self._run_fn(batch, candidate, capture_traces)
        except Exception as exc:  # unrecoverable/systemic
            # Fail the entire batch but keep the engine alive
            outputs = [
                {"status": "error", "error": f"system failure: {exc.__class__.__name__}: {exc}"}
                for _ in batch
            ]
            scores = [0.0 for _ in batch]
            trajectories = [
                {"trace": [], "errors": [str(exc)], "candidate": candidate}
                for _ in batch
            ] if capture_traces else None
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

        outputs: List[AgenticRollout] = []
        scores: List[float] = []
        trajectories: Optional[List[AgenticTrajectory]] = [] if capture_traces else None

        for idx, (out, score, traj) in enumerate(results):
            # robustize per-example failures
            if out is None:
                out = {"status": "error", "error": "null output"}
            if not isinstance(score, (int, float)):
                score = 0.0
            # append
            outputs.append(out)
            scores.append(float(score))
            if capture_traces:
                trajectories.append(traj or {"trace": [], "errors": ["missing trajectory"], "index": idx})

        # Correctness constraints
        if len(outputs) != len(batch) or len(scores) != len(batch):
            # normalize sizes to keep the engine safe
            n = len(batch)
            outputs = outputs[:n] + [{"status": "error", "error": "truncated"}] * max(0, n - len(outputs))
            scores = scores[:n] + [0.0] * max(0, n - len(scores))
            if capture_traces:
                assert trajectories is not None
                trajectories = trajectories[:n] + [{"trace": [], "errors": ["truncated"]}] * max(0, n - len(trajectories))

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # -----------------------------------------
    # 2) Reflective dataset for teacher updates
    # -----------------------------------------
    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[AgenticTrajectory, AgenticRollout],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        # Build concise, high-signal records for each requested component.
        dataset: Dict[str, List[Dict[str, Any]]] = {name: [] for name in components_to_update}

        # Prepare iteration over examples
        num_examples = len(eval_batch.scores)
        trajectories = eval_batch.trajectories or [None] * num_examples

        for ex_idx in range(num_examples):
            score = float(eval_batch.scores[ex_idx])
            output = eval_batch.outputs[ex_idx] if ex_idx < len(eval_batch.outputs) else {}
            traj = trajectories[ex_idx] if ex_idx < len(trajectories) else None

            # Extract minimal Inputs for reflection
            inputs_view = {}
            # If run_fn populates trajectories with config, surface a compact view
            if isinstance(traj, dict):
                # Avoid large blobs; only keep compact keys if present
                for k in ("mode", "tools", "target_image_path", "blender_file", "calls", "errors"):
                    if k in traj:
                        inputs_view[k] = traj[k]

            generated_outputs = output
            feedback = {
                "score": score,
                "summary": _summarize_trajectory(traj),
            }

            # Build per-component records
            for comp in components_to_update:
                record = {
                    "Inputs": inputs_view,
                    "Generated Outputs": generated_outputs,
                    "Feedback": _format_feedback_for_component(comp, candidate.get(comp, ""), feedback),
                    # Helpful extras for teacher prompts
                    "component_text": candidate.get(comp, ""),
                    "component_name": comp,
                    "trace_id": f"ex{ex_idx}:{comp}",
                }
                dataset[comp].append(_prune_large_values(record))

        # Optional: deterministic sub-sampling per component (keep at most N)
        max_per_component = 8
        for comp, records in dataset.items():
            if len(records) > max_per_component:
                self._rng.shuffle(records)
                dataset[comp] = sorted(records[:max_per_component], key=lambda r: r.get("score", 0), reverse=True)

        return dataset

    # ---------------------
    # Default runner (noop)
    # ---------------------
    @staticmethod
    def _default_noop_run(
        batch: List[AgenticDataInst],
        candidate: Dict[str, str],
        capture_traces: bool,
    ) -> List[Tuple[AgenticRollout, float, Optional[AgenticTrajectory]]]:
        """Fallback runner that returns neutral outputs and zero scores.

        Replace this by passing a concrete `run_fn` to the adapter constructor.
        """
        results: List[Tuple[AgenticRollout, float, Optional[AgenticTrajectory]]] = []
        for inst in batch:
            out: AgenticRollout = {
                "status": "noop",
                "message": "Runner not wired. Provide run_fn to execute agents.",
            }
            score = 0.0
            traj: Optional[AgenticTrajectory] = None
            if capture_traces:
                traj = {
                    "mode": inst.mode,
                    "tools": {
                        "generator": inst.generator_tools,
                        "verifier": inst.verifier_tools,
                    },
                    "calls": [],
                    "errors": ["noop-runner"],
                }
            results.append((out, score, traj))
        return results


# -------------------------
# Helper formatting utils
# -------------------------


def _summarize_trajectory(traj: Optional[Dict[str, Any]]) -> str:
    if not traj:
        return "No trajectory captured."
    summary = {
        k: v
        for k, v in traj.items()
        if k in ("mode", "errors") or k.startswith("tool_") or k == "calls"
    }
    try:
        return json.dumps(summary, ensure_ascii=False)[:2000]
    except Exception:
        return str(summary)[:2000]


def _format_feedback_for_component(component_name: str, component_text: str, base_feedback: Dict[str, Any]) -> str:
    parts = [
        f"Component: {component_name}",
        f"Length: {len(component_text or '')}",
        f"Score: {base_feedback.get('score', 0.0)}",
    ]
    if base_feedback.get("summary"):
        parts.append(f"Trace Summary: {base_feedback['summary']}")
    return " | ".join(parts)


def _prune_large_values(record: Dict[str, Any], max_len: int = 5000) -> Dict[str, Any]:
    """Ensure JSON-serializable and limit oversized blobs in-place for safety."""
    def _clip(val: Any) -> Any:
        if isinstance(val, str) and len(val) > max_len:
            return val[:max_len] + " ...[truncated]"
        if isinstance(val, (list, tuple)) and len(val) > 100:
            return list(val)[:100] + ["...[truncated]"]
        if isinstance(val, dict):
            return {k: _clip(v) for k, v in list(val.items())[:50]}
        return val

    return {k: _clip(v) for k, v in record.items()}


