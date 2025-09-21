#!/usr/bin/env python3
"""videocut4.py - Deterministic single-take editing pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


# ---------------------------------------------------------------------------
# Utility helpers


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp()}] {message}\n")


def ensure_ffmpeg_available() -> Tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must be installed and on PATH")
    return ffmpeg, ffprobe


def try_get_openrouter_api_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if platform.system().lower() == "windows":
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, r"Environment") as reg:
                        value, _ = winreg.QueryValueEx(reg, "OPENROUTER_API_KEY")
                        if value:
                            return value
                except OSError:
                    continue
        except ImportError:
            pass
    return None


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise SystemExit(f"Input pattern '{pattern}' matched no files")
        expanded.extend(Path(match).resolve() for match in matches)
    if not expanded:
        raise SystemExit("No input media files resolved")
    return expanded


def compute_stem(paths: Sequence[Path]) -> str:
    first = paths[0]
    joined = "|".join(str(p) for p in paths)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    return f"{first.stem}_{digest}"


def parse_target_length(value: str) -> float:
    if value is None or value == "":
        return 0.0
    value = value.strip()
    if value == "0":
        return 0.0
    if ":" in value:
        parts = value.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            try:
                minutes_f = float(minutes)
                seconds_f = float(seconds)
            except ValueError as exc:
                raise SystemExit(f"Invalid target length '{value}'") from exc
            return max(minutes_f * 60.0 + seconds_f, 0.0)
        raise SystemExit(f"Unsupported target length format '{value}'")
    try:
        seconds = float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid target length '{value}'") from exc
    return max(seconds, 0.0)


def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    for key, val in weights.items():
        if not (0.0 <= val <= 1.0):
            raise SystemExit(f"Score weight '{key}' must be between 0 and 1")
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("Score weights sum must be positive")
    return {key: val / total for key, val in weights.items()}


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed JSON at {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def trim_text(value: str) -> str:
    return value.rstrip()


def seconds_to_mmss(value: float) -> str:
    if value <= 0:
        return "0"
    minutes = int(value // 60)
    seconds = int(round(value - minutes * 60))
    return f"{minutes:02d}:{seconds:02d}"


def ensure_non_empty_segments(segments: Sequence[Dict[str, Any]]) -> None:
    if not segments:
        raise SystemExit("Transcript contains no segments")


def shlex_quote(value: str) -> str:
    if not value:
        return "''"
    if all(c.isalnum() or c in "@%_-+=:,./" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def safe_join_command(command: Sequence[str]) -> str:
    return " ".join(shlex_quote(part) for part in command)


def float_round(value: float) -> float:
    return float(f"{value:.6f}")


def concat_escape(path: Path) -> str:
    return str(path).replace("'", "''")


def format_ffmpeg_time(seconds_value: float) -> str:
    milliseconds = int(round(seconds_value * 1000))
    seconds_float = milliseconds / 1000.0
    return f"{seconds_float:.3f}"


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class ScoreWeights:
    fact: float
    info: float
    form: float
    importance: float


@dataclass
class RunConfig:
    input_paths: List[Path]
    language: Optional[str]
    whisper_model: str
    openrouter_model: str
    weights: ScoreWeights
    fact_threshold: float
    form_threshold: float
    intent_hint: Optional[str]
    target_length: float
    cut_tolerance: Optional[float]
    cache_dir: Path
    output_dir: Path
    plan_only: bool
    debug: bool
    score_batch_size: int = 50


@dataclass
class RunContext:
    config: RunConfig
    stem: str
    run_cache_dir: Path
    run_output_dir: Path
    metadata_path: Path
    transcript_path: Path
    intent_path: Path
    scores_path: Path
    outline_path: Path
    section_plan_path: Path
    optimization_path: Path
    final_video_path: Path
    plan_log_path: Path
    render_log_path: Path
    debug_dir: Path
    ffmpeg_path: str
    ffprobe_path: str


# ---------------------------------------------------------------------------
# Argument parsing


def parse_args(argv: Sequence[str]) -> RunConfig:
    parser = argparse.ArgumentParser(description="Automated single-take video cutter")
    parser.add_argument("--input", nargs="+", required=True, help="Media file paths or globs")
    parser.add_argument("--language", choices=["en", "de"], help="Optional Whisper language hint")
    parser.add_argument("--intent", help="Free-form description of the video's intent")
    parser.add_argument("--cut-tolerance", type=float, help="Desired max cuts per minute")
    parser.add_argument("--fact-threshold", type=float, default=0.0, help="Fact score threshold [-1,1]")
    parser.add_argument("--form-threshold", type=float, default=0.3, help="Form score threshold [0,1]")
    parser.add_argument("--target-length", default="0", help="Desired target length in seconds or mm:ss")
    parser.add_argument("--score-weight-fact", type=float, default=0.25)
    parser.add_argument("--score-weight-info", type=float, default=0.25)
    parser.add_argument("--score-weight-form", type=float, default=0.25)
    parser.add_argument("--score-weight-importance", type=float, default=0.25)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--model", default="openai/gpt-5")

    parsed = parser.parse_args(argv)

    if not (-1.0 <= parsed.fact_threshold <= 1.0):
        raise SystemExit("--fact-threshold must be in [-1, 1]")
    if not (0.0 <= parsed.form_threshold <= 1.0):
        raise SystemExit("--form-threshold must be in [0, 1]")

    resolved_inputs = expand_inputs(parsed.input)
    target_length = parse_target_length(str(parsed.target_length))

    weights = normalise_weights(
        {
            "fact": parsed.score_weight_fact,
            "info": parsed.score_weight_info,
            "form": parsed.score_weight_form,
            "importance": parsed.score_weight_importance,
        }
    )

    config = RunConfig(
        input_paths=resolved_inputs,
        language=parsed.language,
        whisper_model=parsed.whisper_model,
        openrouter_model=parsed.model,
        weights=ScoreWeights(**weights),
        fact_threshold=parsed.fact_threshold,
        form_threshold=parsed.form_threshold,
        intent_hint=parsed.intent,
        target_length=target_length,
        cut_tolerance=parsed.cut_tolerance,
        cache_dir=Path(parsed.cache_dir).resolve(),
        output_dir=Path(parsed.output_dir).resolve(),
        plan_only=parsed.plan_only,
        debug=parsed.debug,
    )
    return config


# ---------------------------------------------------------------------------
# Metadata helpers


def build_context(config: RunConfig) -> RunContext:
    stem = compute_stem(config.input_paths)
    run_cache_dir = config.cache_dir / stem
    run_output_dir = config.output_dir / stem
    context = RunContext(
        config=config,
        stem=stem,
        run_cache_dir=run_cache_dir,
        run_output_dir=run_output_dir,
        metadata_path=run_cache_dir / "metadata.json",
        transcript_path=run_cache_dir / "transcript.json",
        intent_path=run_cache_dir / "intent.md",
        scores_path=run_cache_dir / "scores.json",
        outline_path=run_cache_dir / "outline.json",
        section_plan_path=run_cache_dir / "section_plan.json",
        optimization_path=run_cache_dir / "optimization.json",
        final_video_path=run_output_dir / f"{stem}_cut.mp4",
        plan_log_path=run_cache_dir / "plan.log",
        render_log_path=run_cache_dir / "render.log",
        debug_dir=run_cache_dir / "debug",
        ffmpeg_path="",
        ffprobe_path="",
    )
    return context


def current_metadata_payload(context: RunContext) -> Dict[str, Any]:
    cfg = context.config
    return {
        "timestamp": timestamp(),
        "args": {
            "input_paths": [str(p) for p in cfg.input_paths],
            "language": cfg.language,
            "whisper_model": cfg.whisper_model,
            "openrouter_model": cfg.openrouter_model,
            "weights": dataclasses.asdict(cfg.weights),
            "fact_threshold": cfg.fact_threshold,
            "form_threshold": cfg.form_threshold,
            "intent_hint": cfg.intent_hint,
            "target_length": cfg.target_length,
            "cut_tolerance": cfg.cut_tolerance,
            "plan_only": cfg.plan_only,
            "debug": cfg.debug,
        },
    }


def determine_earliest_stage(context: RunContext, metadata: Optional[Dict[str, Any]]) -> int:
    stage_paths: List[List[Path]] = [
        [context.transcript_path],
        [context.intent_path],
        [context.scores_path],
        [context.outline_path],
        [context.section_plan_path],
        [context.optimization_path],
    ]
    if not context.config.plan_only:
        stage_paths.append([context.final_video_path])

    if metadata is None:
        return 0

    if metadata.get("args") != current_metadata_payload(context)["args"]:
        return 0

    for idx, paths in enumerate(stage_paths):
        if any(not path.exists() for path in paths):
            return idx
    return len(stage_paths)


# ---------------------------------------------------------------------------
# Stage helpers


def load_transcript_if_exists(context: RunContext) -> Optional[Dict[str, Any]]:
    if not context.transcript_path.exists():
        return None
    return read_json(context.transcript_path)


def run_transcript_stage(context: RunContext) -> Dict[str, Any]:
    cfg = context.config
    write_log(context.plan_log_path, "Stage A: Transcript harvest start")

    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency check
        raise SystemExit("Missing dependency: openai-whisper is required") from exc

    model = whisper.load_model(cfg.whisper_model)

    files_table = []
    all_segments: List[Dict[str, Any]] = []

    for file_id, path in enumerate(cfg.input_paths):
        files_table.append({"file_id": file_id, "path": str(path)})
        transcript_name = path.with_suffix(".json").name
        preferred_path = path.parent / transcript_name

        transcript_data: Optional[Dict[str, Any]] = None
        if preferred_path.exists():
            transcript_data = read_json(preferred_path)
        else:
            try:
                result = model.transcribe(
                    str(path),
                    language=cfg.language,
                    word_timestamps=True,
                )
            except Exception as exc:  # pragma: no cover - external process
                raise SystemExit(f"Whisper transcription failed for {path}: {exc}") from exc

            segments = []
            for segment in result.get("segments", []):
                words = segment.get("words") or []
                if not words:
                    continue
                start = float(words[0].get("start", segment.get("start", 0.0)))
                end = float(words[-1].get("end", segment.get("end", 0.0)))
                text = segment.get("text", "").strip()
                segments.append({"start": float_round(start), "end": float_round(end), "text": text})

            ensure_non_empty_segments(segments)

            filename_value = path.name if os.access(path.parent, os.W_OK) else str(path)
            transcript_data = {"filename": filename_value, "segments": segments}

            try:
                with preferred_path.open("w", encoding="utf-8") as fh:
                    json.dump(transcript_data, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
            except OSError:
                cache_target = context.run_cache_dir / transcript_name
                with cache_target.open("w", encoding="utf-8") as fh:
                    json.dump(transcript_data, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")

        if not transcript_data:
            raise SystemExit(f"Transcript missing for {path}")

        for segment in transcript_data.get("segments", []):
            segment_id = len(all_segments)
            all_segments.append(
                {
                    "segment_id": segment_id,
                    "file_id": file_id,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": segment["text"],
                }
            )

    ensure_non_empty_segments(all_segments)

    payload = {"files": files_table, "segments": all_segments}
    write_json(context.transcript_path, payload)
    write_log(context.plan_log_path, f"Stage A: Transcript harvest complete ({len(all_segments)} segments)")
    return payload


def load_intent_if_exists(context: RunContext) -> Optional[str]:
    if not context.intent_path.exists():
        return None
    with context.intent_path.open("r", encoding="utf-8") as fh:
        return fh.read()


def call_openrouter(context: RunContext, tag: str, messages: List[Dict[str, str]], temperature: float) -> str:
    api_key = try_get_openrouter_api_key()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for GPT stages")

    payload = {
        "model": context.config.openrouter_model,
        "messages": messages,
        "temperature": temperature,
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if context.config.debug:
        debug_stage_dir = context.debug_dir / tag
        debug_stage_dir.mkdir(parents=True, exist_ok=True)
        request_path = debug_stage_dir / "request.json"
        with request_path.open("w", encoding="utf-8") as fh:
            json.dump({"url": url, "payload": payload}, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        response_path = debug_stage_dir / "response.txt"
        with response_path.open("w", encoding="utf-8") as fh:
            fh.write(response.text)

    if not (200 <= response.status_code < 300):
        snippet = response.text[:4000]
        raise SystemExit(f"OpenRouter request failed ({response.status_code}): {snippet}")

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON from OpenRouter: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit("Unexpected OpenRouter response schema") from exc

    return content


def run_intent_stage(context: RunContext, transcript: Dict[str, Any]) -> str:
    write_log(context.plan_log_path, "Stage B: Intent planning start")

    transcript_text = "\n".join(segment["text"] for segment in transcript["segments"])
    hint = context.config.intent_hint.strip() if context.config.intent_hint else "None provided."

    system_prompt = (
        "You are an editorial strategist distilling a single-take recording. Given the transcript and user hint, summarise the "
        "video's intent and organise key themes. Keep the response under roughly 250 words."
    )

    user_prompt = textwrap.dedent(
        f"""
        Transcript:
        {transcript_text}

        User intent hint:
        {hint}

        Produce:
        - 2–3 bullets covering purpose, audience, and tone.
        - Four headings (**Essential**, **Important**, **Optional**, **Leave out**) each with concise thematic bullets.
        Avoid referencing segment IDs or timings.
        """
    ).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = call_openrouter(context, "intent", messages, temperature=0.0)
    content = trim_text(content)
    write_log(context.plan_log_path, "Stage B: Intent planning complete")
    with context.intent_path.open("w", encoding="utf-8") as fh:
        fh.write(content + "\n")
    return content


def load_scores_if_exists(context: RunContext) -> Optional[Dict[str, Any]]:
    if not context.scores_path.exists():
        return None
    return read_json(context.scores_path)


def batch(iterable: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def run_scoring_stage(context: RunContext, transcript: Dict[str, Any]) -> Dict[str, Any]:
    write_log(context.plan_log_path, "Stage C: Scoring start")
    segments = transcript["segments"]
    annotated_segments: List[Dict[str, Any]] = []
    annotated_lookup: Dict[int, Dict[str, Any]] = {}
    for segment in segments:
        duration = max(float(segment["end"]) - float(segment["start"]), 1e-6)
        speed = len(segment["text"]) / duration
        item = {
            "segment_id": segment["segment_id"],
            "text": segment["text"],
            "duration": float_round(duration),
            "speed": float_round(speed),
        }
        annotated_segments.append(item)
        annotated_lookup[item["segment_id"]] = item

    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []

    weights = context.config.weights
    for sub_segments in batch(annotated_segments, context.config.score_batch_size):
        system_prompt = textwrap.dedent(
            """
            You are scoring transcript segments for editorial selection. Respond with strict JSON matching
            {"segment_scores": [{"segment_id": int, "fact": float, "info": float, "form": float, "importance": float, "note": "..."}]}.
            Rubric:
            - fact: +1 truthful/precise, 0 neutral/uncertain, negative if incorrect or misleading.
            - info: 1 adds substantive new information toward the intent, 0 if filler.
            - form: 1 for clear delivery, coherent sentences, and pacing. Use supplied speed (chars/sec). Target 12–18 cps (12–16 fine for dense or slower styles); soft penalty at 10–12 or 18–20; strong penalty below 10 or above 20.
            - importance: 1 if critical for the outlined intent, proportionally lower otherwise.
            Provide a short justification note for each segment.
            Temperature must remain zero; do not invent schema changes.
            """
        ).strip()

        batch_payload = [
            {
                "segment_id": item["segment_id"],
                "text": item["text"],
                "duration": item["duration"],
                "speed": item["speed"],
            }
            for item in sub_segments
        ]

        user_prompt = json.dumps(batch_payload, ensure_ascii=False)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content = call_openrouter(context, "scoring", messages, temperature=0.0)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Scoring response was not valid JSON: {exc}\n{content}") from exc

        entries = parsed.get("segment_scores")
        if not isinstance(entries, list):
            raise SystemExit("Scoring response missing 'segment_scores'")

        for entry in entries:
            try:
                segment_id = entry["segment_id"]
                fact = float(entry["fact"])
                info = float(entry["info"])
                form = float(entry["form"])
                importance = float(entry["importance"])
                note = str(entry["note"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid scoring entry: {entry}") from exc

            if segment_id not in annotated_lookup:
                raise SystemExit(f"Scoring response referenced unknown segment {segment_id}")
            source = annotated_lookup[segment_id]
            composite = (
                weights.fact * fact
                + weights.info * info
                + weights.form * form
                + weights.importance * importance
            )
            scored = {
                "segment_id": segment_id,
                "text": source["text"],
                "duration": source["duration"],
                "speed": source["speed"],
                "scores": {
                    "fact": fact,
                    "info": info,
                    "form": form,
                    "importance": importance,
                    "note": note,
                },
                "composite_score": float_round(composite),
            }
            if fact >= context.config.fact_threshold and form >= context.config.form_threshold:
                kept.append(scored)
            else:
                discarded.append(scored)

    payload = {
        "kept": kept,
        "discarded": discarded,
        "weights": dataclasses.asdict(weights),
        "fact_threshold": context.config.fact_threshold,
        "form_threshold": context.config.form_threshold,
    }
    write_json(context.scores_path, payload)

    write_log(
        context.plan_log_path,
        f"Stage C: Scoring complete ({len(kept)} kept, {len(discarded)} discarded)",
    )
    if not kept:
        raise SystemExit("All segments were discarded by thresholds; nothing to plan")
    return payload


def load_outline_if_exists(context: RunContext) -> Optional[List[Dict[str, Any]]]:
    if not context.outline_path.exists():
        return None
    data = read_json(context.outline_path)
    if not isinstance(data, list):
        raise SystemExit("outline.json must be a JSON array")
    return data


def run_outline_stage(
    context: RunContext,
    intent_text: str,
    scores: Dict[str, Any],
) -> List[Dict[str, Any]]:
    write_log(context.plan_log_path, "Stage D: Structure planning start")

    kept_segments = scores["kept"]
    segment_samples = [
        {
            "text": item["text"],
            "duration": item["duration"],
            "speed": item["speed"],
            "composite_score": item["composite_score"],
        }
        for item in kept_segments
    ]

    system_prompt = (
        "You are a senior story editor crafting a broadcast-ready outline. Design sections that flow logically "
        "(intro → core → wrap-up) while respecting pacing hints from duration, speed, and composite scores."
    )
    target_runtime_desc = seconds_to_mmss(context.config.target_length)
    user_payload = {
        "intent": intent_text,
        "target_runtime_seconds": context.config.target_length,
        "target_runtime_label": target_runtime_desc,
        "segments": segment_samples,
    }
    user_prompt = textwrap.dedent(
        """
        Using the provided intent summary and segment samples, design an ordered outline. Output JSON as
        [{"section_id": 1, "title": "...", "goal": "...", "target_duration": float}, ...].
        Section IDs must start at 1 and be contiguous. Include goals that cover transitions (e.g., problem → solution).
        Keep the outline concise and actionable.
        Context:
        {context_json}
        """
    ).strip().format(context_json=json.dumps(user_payload, ensure_ascii=False))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = call_openrouter(context, "outline", messages, temperature=0.0)
    try:
        outline = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Outline response invalid JSON: {exc}\n{content}") from exc

    if not isinstance(outline, list) or not outline:
        raise SystemExit("Outline must be a non-empty JSON array")

    expected_id = 1
    for entry in outline:
        if entry.get("section_id") != expected_id:
            raise SystemExit("Outline section IDs must be contiguous starting at 1")
        expected_id += 1

    write_json(context.outline_path, outline)
    write_log(context.plan_log_path, f"Stage D: Structure planning complete ({len(outline)} sections)")
    return outline


def load_section_plan_if_exists(context: RunContext) -> Optional[List[Dict[str, Any]]]:
    if not context.section_plan_path.exists():
        return None
    data = read_json(context.section_plan_path)
    if not isinstance(data, list):
        raise SystemExit("section_plan.json must be a JSON array")
    return data


def cut_tolerance_text(value: Optional[float]) -> str:
    if value is None:
        return (
            "Every non-sequential jump between chosen segment IDs introduces a visible cut. Cuts are unrestricted for this run."
        )
    return (
        "Every non-sequential jump between chosen segment IDs introduces a visible cut. "
        f"Try to keep visible cuts ≤ {value} per minute of final runtime."
    )


def run_section_plan_stage(
    context: RunContext,
    outline: List[Dict[str, Any]],
    scores: Dict[str, Any],
) -> List[Dict[str, Any]]:
    write_log(context.plan_log_path, "Stage E: Section mapping start")

    kept_segments = scores["kept"]
    segment_list = [
        {
            "segment_id": item["segment_id"],
            "text": item["text"],
            "duration": item["duration"],
            "speed": item["speed"],
            "composite_score": item["composite_score"],
        }
        for item in kept_segments
    ]

    payload = {
        "outline": outline,
        "segments": segment_list,
        "target_runtime_seconds": context.config.target_length,
        "cut_tolerance": context.config.cut_tolerance,
        "cut_guidance": cut_tolerance_text(context.config.cut_tolerance),
    }

    system_prompt = (
        "You are mapping strong transcript segments into outline sections. Assign segment IDs to each section in order, "
        "prioritising narrative flow and intent coverage. Visible cuts are jumps between non-sequential segment IDs; "
        "treat the provided tolerance as a guardrail, not an absolute rule."
    )

    user_prompt = textwrap.dedent(
        """
        For each outline section, choose and order segment IDs that satisfy the goal. Output JSON as
        [{"section_id": int, "segment_ids": [int, ...], "section_notes": "..."}, ...], ordered like the outline.
        Prefer natural adjacency but reach for later segments when necessary to cover goals. Mention trade-offs in section_notes.
        Context:
        {context_json}
        """
    ).strip().format(context_json=json.dumps(payload, ensure_ascii=False))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = call_openrouter(context, "section_plan", messages, temperature=0.0)
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Section plan invalid JSON: {exc}\n{content}") from exc

    if not isinstance(plan, list) or len(plan) != len(outline):
        raise SystemExit("Section plan must match outline length")

    for expected, entry in enumerate(plan, start=1):
        if entry.get("section_id") != expected:
            raise SystemExit("Section plan IDs must align with outline")
        if not isinstance(entry.get("segment_ids"), list):
            raise SystemExit("Each section must include a segment_ids list")

    write_json(context.section_plan_path, plan)
    write_log(context.plan_log_path, "Stage E: Section mapping complete")
    return plan


def flatten_plan(
    plan: List[Dict[str, Any]],
    segment_lookup: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for section in plan:
        section_id = section["section_id"]
        for segment_id in section.get("segment_ids", []):
            if segment_id not in segment_lookup:
                raise SystemExit(f"Section plan references unknown segment {segment_id}")
            data = segment_lookup[segment_id]
            flattened.append(
                {
                    "segment_id": segment_id,
                    "section_id": section_id,
                    "text": data["text"],
                    "duration": data["duration"],
                    "speed": data["speed"],
                    "composite_score": data["composite_score"],
                }
            )
    return flattened


def compute_visible_cuts(segment_ids: Sequence[int]) -> int:
    if not segment_ids:
        return 0
    cuts = 0
    for prev, curr in zip(segment_ids, segment_ids[1:]):
        if curr != prev + 1:
            cuts += 1
    return cuts


def run_optimization_stage(
    context: RunContext,
    intent_text: str,
    outline: List[Dict[str, Any]],
    section_plan: List[Dict[str, Any]],
    scores: Dict[str, Any],
    transcript: Dict[str, Any],
) -> Dict[str, Any]:
    write_log(context.plan_log_path, "Stage F: Optimization start")

    kept_lookup = {item["segment_id"]: item for item in scores["kept"]}
    flattened = flatten_plan(section_plan, kept_lookup)
    segment_ids = [item["segment_id"] for item in flattened]
    total_duration = sum(item["duration"] for item in flattened)
    cuts = compute_visible_cuts(segment_ids)
    runtime_minutes = total_duration / 60.0 if total_duration else 1.0
    cuts_per_minute = cuts / runtime_minutes if runtime_minutes > 0 else 0.0

    status_line = (
        f"Current plan runtime {total_duration:.2f}s (~{seconds_to_mmss(total_duration)}). "
        f"Visible cuts: {cuts} ({cuts_per_minute:.2f}/min). Target runtime {seconds_to_mmss(context.config.target_length)}. "
        f"Cut guidance: {cut_tolerance_text(context.config.cut_tolerance)}"
    )

    prompt_payload = {
        "intent": intent_text,
        "outline": outline,
        "section_plan": section_plan,
        "status": status_line,
        "target_runtime_seconds": context.config.target_length,
        "cut_guidance": cut_tolerance_text(context.config.cut_tolerance),
        "segments": flattened,
    }

    system_prompt = (
        "You are the final editor preparing a broadcast-ready cut. Read the plan like a finished script and make high-impact "
        "adjustments to maximise clarity, emotional flow, and informational value. Use numeric stats as sanity checks only."
    )

    user_prompt = textwrap.dedent(
        """
        Optimise the cut order and selection. Output JSON as
        {"segment_ids": [int, ...], "summary": "..."}.
        At least one segment must remain, and explain key editorial decisions plus runtime/cut changes.
        Context:
        {context_json}
        """
    ).strip().format(context_json=json.dumps(prompt_payload, ensure_ascii=False))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = call_openrouter(context, "optimization", messages, temperature=0.0)
    try:
        optimization = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Optimization response invalid JSON: {exc}\n{content}") from exc

    segment_ids = optimization.get("segment_ids")
    if not isinstance(segment_ids, list) or not segment_ids:
        raise SystemExit("Optimizer returned an empty segment list")

    for segment_id in segment_ids:
        if segment_id not in kept_lookup:
            raise SystemExit(f"Optimizer referenced unknown segment {segment_id}")

    summary = optimization.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise SystemExit("Optimizer must include a textual summary")

    write_json(context.optimization_path, optimization)
    write_log(context.plan_log_path, "Stage F: Optimization complete")
    return optimization


def run_command(command: Sequence[str], log_path: Path) -> Tuple[str, str]:
    joined = safe_join_command(command)
    write_log(log_path, f"RUN {joined}")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    if stdout:
        for line in stdout.strip().splitlines():
            write_log(log_path, f"STDOUT {line}")
    if stderr:
        for line in stderr.strip().splitlines():
            write_log(log_path, f"STDERR {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")
    return stdout, stderr


def run_render_stage(
    context: RunContext,
    optimization: Dict[str, Any],
    transcript: Dict[str, Any],
) -> Path:
    write_log(context.plan_log_path, "Stage G: Rendering start")

    segment_ids = optimization.get("segment_ids")
    if not isinstance(segment_ids, list) or not segment_ids:
        raise SystemExit("Cannot render without segment selections")

    files_lookup = {entry["file_id"]: entry["path"] for entry in transcript.get("files", [])}
    segments_lookup = {entry["segment_id"]: entry for entry in transcript.get("segments", [])}

    context.run_output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = context.run_output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: List[Path] = []
    for idx, segment_id in enumerate(segment_ids, start=1):
        if segment_id not in segments_lookup:
            raise SystemExit(f"Optimization references unknown segment {segment_id}")
        segment = segments_lookup[segment_id]
        file_id = segment["file_id"]
        if file_id not in files_lookup:
            raise SystemExit(f"Missing file mapping for segment {segment_id}")
        src_path = Path(files_lookup[file_id])
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            raise SystemExit(f"Segment {segment_id} has non-positive duration")
        out_path = segments_dir / f"{idx:04d}.mp4"
        if out_path.exists():
            out_path.unlink()

        command = [
            context.ffmpeg_path,
            "-y",
            "-i",
            str(src_path),
            "-ss",
            format_ffmpeg_time(start),
            "-to",
            format_ffmpeg_time(end),
            "-c",
            "copy",
            str(out_path),
        ]
        try:
            run_command(command, context.render_log_path)
            size = out_path.stat().st_size if out_path.exists() else 0
            if size < 1024:
                raise RuntimeError("Segment too small, re-encoding")
        except RuntimeError:
            if out_path.exists():
                out_path.unlink()
            fallback_cmd = [
                context.ffmpeg_path,
                "-y",
                "-i",
                str(src_path),
                "-ss",
                format_ffmpeg_time(start),
                "-to",
                format_ffmpeg_time(end),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_path),
            ]
            run_command(fallback_cmd, context.render_log_path)
        if not out_path.exists() or out_path.stat().st_size <= 0:
            raise SystemExit(f"FFmpeg produced empty segment for {segment_id}")
        extracted_paths.append(out_path)

    concat_list = context.run_output_dir / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as fh:
        for path in extracted_paths:
            fh.write(f"file '{concat_escape(path)}'\n")

    context.final_video_path.parent.mkdir(parents=True, exist_ok=True)
    if context.final_video_path.exists():
        context.final_video_path.unlink()

    copy_cmd = [
        context.ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(context.final_video_path),
    ]
    try:
        run_command(copy_cmd, context.render_log_path)
        size = context.final_video_path.stat().st_size if context.final_video_path.exists() else 0
        if size < 2048:
            raise RuntimeError("Rendered file unexpectedly small")
    except (RuntimeError, OSError):
        if context.final_video_path.exists():
            context.final_video_path.unlink()
        fallback_cmd = [
            context.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(context.final_video_path),
        ]
        run_command(fallback_cmd, context.render_log_path)

    if not context.final_video_path.exists() or context.final_video_path.stat().st_size <= 0:
        raise SystemExit("Final render is empty")

    write_log(context.plan_log_path, "Stage G: Rendering complete")
    print(str(context.final_video_path))
    return context.final_video_path


# ---------------------------------------------------------------------------
# Pipeline execution


def run_pipeline(config: RunConfig) -> None:
    ffmpeg_path, ffprobe_path = ensure_ffmpeg_available()
    context = build_context(config)
    context.ffmpeg_path = ffmpeg_path
    context.ffprobe_path = ffprobe_path

    context.run_cache_dir.mkdir(parents=True, exist_ok=True)
    context.run_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(context.metadata_path) if context.metadata_path.exists() else None
    earliest = determine_earliest_stage(context, metadata)

    # Stage A
    if earliest <= 0:
        transcript = run_transcript_stage(context)
    else:
        transcript = load_transcript_if_exists(context) or run_transcript_stage(context)

    # Stage B
    if earliest <= 1:
        intent_text = run_intent_stage(context, transcript)
    else:
        intent_text = load_intent_if_exists(context) or run_intent_stage(context, transcript)

    # Stage C
    if earliest <= 2:
        scores = run_scoring_stage(context, transcript)
    else:
        scores = load_scores_if_exists(context) or run_scoring_stage(context, transcript)

    # Stage D
    if earliest <= 3:
        outline = run_outline_stage(context, intent_text, scores)
    else:
        outline = load_outline_if_exists(context) or run_outline_stage(context, intent_text, scores)

    # Stage E
    if earliest <= 4:
        section_plan = run_section_plan_stage(context, outline, scores)
    else:
        section_plan = load_section_plan_if_exists(context) or run_section_plan_stage(context, outline, scores)

    # Stage F
    if earliest <= 5:
        optimization = run_optimization_stage(context, intent_text, outline, section_plan, scores, transcript)
    else:
        optimization = read_json(context.optimization_path)

    write_json(context.metadata_path, current_metadata_payload(context))

    if context.config.plan_only:
        print(str(context.optimization_path))
        return

    # Stage G
    if earliest <= 6 or not context.final_video_path.exists():
        run_render_stage(context, optimization, transcript)
    else:
        # Everything up to date; emit final video path for convenience
        print(str(context.final_video_path))


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv if argv is not None else sys.argv[1:])
    run_pipeline(config)


if __name__ == "__main__":
    main()
