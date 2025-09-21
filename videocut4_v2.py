#!/usr/bin/env python3
"""videocut4.py

Command line tool described in ``videocut4.md``.  The implementation follows the
specification closely and organises the pipeline into deterministic, cache-first
stages.  Each stage produces well-defined artefacts and aborts early on
inconsistencies or missing dependencies.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


class PipelineError(SystemExit):
    """Raised when the pipeline encounters a fatal condition."""


def abort(message: str) -> None:
    """Raise ``PipelineError`` with a formatted message."""

    raise PipelineError(message)


def ensure_ffmpeg_available() -> None:
    """Validate ``ffmpeg`` and ``ffprobe`` are on PATH."""

    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        abort(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install FFmpeg before running the pipeline."
        )


def try_get_openrouter_api_key() -> Optional[str]:
    """Return an API key from environment variables or Windows registry."""

    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    if os.name == "nt":  # pragma: no cover - Windows only path
        try:
            import winreg  # type: ignore

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, r"Environment") as reg_key:
                        value, _ = winreg.QueryValueEx(reg_key, "OPENROUTER_API_KEY")
                        if value:
                            return str(value)
                except FileNotFoundError:
                    continue
        except ImportError:
            pass

    return None


def append_log(path: Path, message: str) -> None:
    """Append a timestamped message to ``path``."""

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def run_subprocess(command: Sequence[str], log_path: Optional[Path] = None) -> Tuple[int, str, str]:
    """Execute a command returning ``(return_code, stdout, stderr)``."""

    if log_path is not None:
        append_log(log_path, "Executing: " + " ".join(shlex.quote(part) for part in command))

    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def normalise_score_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        abort("Score weights must sum to a positive value.")
    return {name: value / total for name, value in weights.items()}


def parse_target_length(value: Optional[str]) -> float:
    if not value:
        return 0.0

    value = value.strip()
    mmss = re.fullmatch(r"(\d+):(\d{2})", value)
    if mmss:
        minutes = int(mmss.group(1))
        seconds = int(mmss.group(2))
        if seconds >= 60:
            abort("Target length mm:ss string must have seconds < 60.")
        return float(minutes * 60 + seconds)

    try:
        result = float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        abort(f"Invalid target length '{value}': {exc}")
    if result < 0:
        abort("Target length must be non-negative.")
    return result


def expand_input_paths(patterns: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            abort(f"Input pattern '{pattern}' produced no matches.")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                abort(f"Input path '{path}' is not a file.")
            expanded.append(path)
    if not expanded:
        abort("No input media files resolved.")
    return expanded


def compute_stem(paths: Sequence[Path]) -> str:
    first = paths[0].name
    joined = "\n".join(str(p) for p in paths)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    return f"{first}_{digest}"


def load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        abort(f"Failed to read JSON file '{path}': {exc}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        abort(f"Invalid JSON in '{path}': {exc}")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        abort(f"Failed to write JSON to '{path}': {exc}")


def dump_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(data, encoding="utf-8")
    except OSError as exc:
        abort(f"Failed to write text to '{path}': {exc}")


def gather_transcript_segments(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "segments" not in transcript or not isinstance(transcript["segments"], list):
        abort("Transcript JSON missing 'segments' list.")
    segments: List[Dict[str, Any]] = []
    for index, raw_segment in enumerate(transcript["segments"]):
        try:
            start = float(raw_segment["start"])
            end = float(raw_segment["end"])
            text = str(raw_segment["text"])
        except (KeyError, TypeError, ValueError) as exc:
            abort(f"Invalid segment at index {index}: {exc}")
        if end < start:
            abort(f"Segment {index} has end < start.")
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        abort("Transcript contains no segments.")
    return segments


@dataclass
class PipelineContext:
    args: argparse.Namespace
    input_paths: List[Path]
    stem: str
    cache_root: Path
    run_cache_dir: Path
    output_root: Path
    run_output_dir: Path
    metadata_path: Path
    plan_log: Path
    render_log: Path
    debug_root: Path
    weights: Dict[str, float]
    target_length: float
    cut_tolerance: Optional[float]
    api_key: Optional[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, Dict[str, float], float]:
    parser = argparse.ArgumentParser(description="Automated video cut planner")
    parser.add_argument("--input", nargs="+", required=True, help="Media input paths (globs allowed)")
    parser.add_argument("--language", choices=["en", "de"], help="Whisper language hint", default=None)
    parser.add_argument("--intent", default=None, help="Free-form text describing the desired outcome")
    parser.add_argument("--cut-tolerance", type=float, default=None, help="Maximum visible cuts per minute")
    parser.add_argument("--fact-threshold", type=float, default=0.0)
    parser.add_argument("--form-threshold", type=float, default=0.3)
    parser.add_argument("--target-length", default=None)
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

    args = parser.parse_args(argv)

    if not (-1.0 <= args.fact_threshold <= 1.0):
        abort("--fact-threshold must be within [-1, 1].")
    if not (0.0 <= args.form_threshold <= 1.0):
        abort("--form-threshold must be within [0, 1].")

    weights = {
        "fact": args.score_weight_fact,
        "info": args.score_weight_info,
        "form": args.score_weight_form,
        "importance": args.score_weight_importance,
    }
    for name, value in weights.items():
        if value < 0 or value > 1:
            abort(f"Score weight '{name}' must be within [0, 1].")
    weights = normalise_score_weights(weights)

    target_length = parse_target_length(args.target_length)

    return args, weights, target_length


def stage_requires_gpt(index: int) -> bool:
    """Return ``True`` for stages B–F."""

    return index in {1, 2, 3, 4, 5}


def determine_recompute_point(
    ctx: PipelineContext, metadata_payload: Dict[str, Any]
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Determine earliest stage needing recomputation."""

    recompute_from: Optional[int] = None
    previous: Optional[Dict[str, Any]] = None
    if ctx.metadata_path.exists():
        previous = load_json(ctx.metadata_path)
        if not isinstance(previous, dict) or "args" not in previous:
            abort("metadata.json has unexpected structure.")

    if previous is None or previous.get("args") != metadata_payload.get("args"):
        recompute_from = 0

    stage_artifacts: List[Tuple[int, Path]] = [
        (0, ctx.run_cache_dir / "transcript.json"),
        (1, ctx.run_cache_dir / "intent.md"),
        (2, ctx.run_cache_dir / "scores.json"),
        (3, ctx.run_cache_dir / "outline.json"),
        (4, ctx.run_cache_dir / "section_plan.json"),
        (5, ctx.run_cache_dir / "optimization.json"),
    ]
    for index, path in stage_artifacts:
        if not path.exists():
            if recompute_from is None or index < recompute_from:
                recompute_from = index

    if not ctx.args.plan_only:
        output_path = ctx.run_output_dir / f"{ctx.stem}_cut.mp4"
        if not output_path.exists():
            if recompute_from is None or 6 < recompute_from:
                recompute_from = 6

    return recompute_from, previous or {}


def ensure_dependencies(ctx: PipelineContext, recompute_from: Optional[int]) -> None:
    ensure_ffmpeg_available()
    if recompute_from is not None:
        needs_gpt = any(stage_requires_gpt(idx) and idx >= recompute_from for idx in range(7))
        if needs_gpt:
            ctx.api_key = try_get_openrouter_api_key()
            if not ctx.api_key:
                abort("OPENROUTER_API_KEY is required for GPT stages but was not found.")


def load_or_run_stage_a(ctx: PipelineContext, should_run: bool) -> Dict[str, Any]:
    transcript_path = ctx.run_cache_dir / "transcript.json"
    if not should_run and transcript_path.exists():
        return load_json(transcript_path)

    append_log(ctx.plan_log, "Running Stage A – Transcript Harvest")

    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency error
        abort(f"Stage A requires the 'whisper' package: {exc}")

    model = whisper.load_model(ctx.args.whisper_model)

    files_entries: List[Dict[str, Any]] = []
    segments_entries: List[Dict[str, Any]] = []
    segment_counter = 0

    for file_id, media_path in enumerate(ctx.input_paths):
        candidate = media_path.with_suffix(".json")
        if candidate.exists():
            transcript_data = load_json(candidate)
            append_log(ctx.plan_log, f"Using cached transcript '{candidate}'.")
        else:
            append_log(ctx.plan_log, f"Transcribing '{media_path}'.")
            result = model.transcribe(
                str(media_path),
                language=ctx.args.language,
                word_timestamps=True,
            )
            transcript_segments = []
            for seg in result.get("segments", []):
                words = seg.get("words") or []
                if words:
                    start = float(words[0]["start"])
                    end = float(words[-1]["end"])
                else:
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", start))
                text = str(seg.get("text", "")).strip()
                transcript_segments.append({"start": start, "end": end, "text": text})
            if not transcript_segments:
                abort(f"Whisper returned no segments for '{media_path}'.")

            transcript_data = {"filename": media_path.name, "segments": transcript_segments}

            target_path = candidate
            try:
                with target_path.open("w", encoding="utf-8") as handle:
                    json.dump(transcript_data, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
            except OSError:
                append_log(
                    ctx.plan_log,
                    f"Unable to write transcript next to media. Storing in cache for '{media_path}'.",
                )
                transcript_data["filename"] = str(media_path.resolve())
                target_path = ctx.run_cache_dir / f"{media_path.stem}.json"
                dump_json(target_path, transcript_data)
            else:
                append_log(ctx.plan_log, f"Stored transcript at '{target_path}'.")

        segments = gather_transcript_segments(transcript_data)
        files_entries.append({"file_id": file_id, "path": str(media_path)})
        for segment in segments:
            entry = {
                "segment_id": segment_counter,
                "file_id": file_id,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"],
            }
            segments_entries.append(entry)
            segment_counter += 1

    if not segments_entries:
        abort("No transcript segments found across inputs.")

    payload = {"files": files_entries, "segments": segments_entries}
    dump_json(transcript_path, payload)
    append_log(ctx.plan_log, f"Wrote transcript.json with {segment_counter} segments.")
    return payload


def gather_transcript_text(transcript: Dict[str, Any]) -> str:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        abort("transcript.json missing 'segments' array.")
    return "\n".join(str(segment.get("text", "")) for segment in segments)


def prepare_gpt_payload(
    ctx: PipelineContext,
    stage: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    return {
        "model": ctx.args.model,
        "messages": messages,
        "temperature": temperature,
    }


def call_openrouter(
    ctx: PipelineContext,
    stage: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> str:
    if not ctx.api_key:
        abort("OPENROUTER_API_KEY not set; cannot perform GPT call.")

    payload = prepare_gpt_payload(ctx, stage, messages, temperature)
    headers = {
        "Authorization": f"Bearer {ctx.api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:  # pragma: no cover - network error
        abort(f"OpenRouter request for stage '{stage}' failed: {exc}")
    if ctx.args.debug:
        stage_dir = ctx.debug_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        dump_json(stage_dir / "request.json", payload)
        try:
            dump_text(stage_dir / "response.txt", response.text)
        except Exception:  # pragma: no cover - defensive
            pass

    if not response.ok:
        body_preview = response.text[:4000]
        abort(
            f"OpenRouter request for stage '{stage}' failed with status {response.status_code}: {body_preview}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        abort(f"OpenRouter response for stage '{stage}' is not valid JSON: {exc}")

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        abort(f"Unexpected OpenRouter response structure for stage '{stage}': {exc}")


def run_stage_b_intent(
    ctx: PipelineContext,
    transcript: Dict[str, Any],
    should_run: bool,
) -> str:
    path = ctx.run_cache_dir / "intent.md"
    if not should_run and path.exists():
        return path.read_text(encoding="utf-8")

    append_log(ctx.plan_log, "Running Stage B – Intent")

    transcript_text = gather_transcript_text(transcript)
    intent_hint = ctx.args.intent.strip() if ctx.args.intent else "None provided."

    system_prompt = (
        "You are an experienced post-production story editor. Given a raw transcript, "
        "summarise the project's purpose, audience, and tone, then organise content "
        "themes by priority. Keep the response under approximately 250 words and avoid "
        "referencing segment IDs or timestamps."
    )

    user_prompt = (
        "Transcript:\n"
        f"{transcript_text}\n\n"
        "User intent hint:\n"
        f"{intent_hint}\n\n"
        "Respond in Markdown with:\n"
        "- 2–3 bullet points covering purpose, audience, and tone.\n"
        "- Headings **Essential**, **Important**, **Optional**, **Leave out** each with concise bullets of themes."
    )

    response = call_openrouter(
        ctx,
        stage="intent",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )

    response = response.rstrip()
    dump_text(path, response)
    append_log(ctx.plan_log, "Stored intent.md")
    return response


def compute_segment_stats(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        abort("transcript.json missing 'segments'.")
    stats: List[Dict[str, Any]] = []
    for seg in segments:
        try:
            segment_id = int(seg["segment_id"])
            start = float(seg["start"])
            end = float(seg["end"])
            text = str(seg["text"])
        except (KeyError, TypeError, ValueError) as exc:
            abort(f"Invalid segment in transcript.json: {exc}")
        duration = max(end - start, 0.0)
        speed = len(text) / max(duration, 1e-6)
        stats.append(
            {
                "segment_id": segment_id,
                "text": text,
                "duration": duration,
                "speed": speed,
            }
        )
    return stats


def run_stage_c_scoring(
    ctx: PipelineContext,
    segments_stats: List[Dict[str, Any]],
    should_run: bool,
) -> Dict[str, Any]:
    path = ctx.run_cache_dir / "scores.json"
    if not should_run and path.exists():
        return load_json(path)

    append_log(ctx.plan_log, "Running Stage C – Scoring")

    batches: List[List[Dict[str, Any]]] = []
    batch_size = 50
    for i in range(0, len(segments_stats), batch_size):
        batches.append(segments_stats[i : i + batch_size])

    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    threshold_stats = {"fact": 0, "form": 0}

    for batch_index, batch in enumerate(batches):
        system_prompt = (
            "You are an editorial scoring assistant. Score each segment according to the "
            "rubric.\n"
            "fact: +1 truthful/precise, 0 neutral/uncertain, negative for incorrect.\n"
            "info: 1 if the segment adds substantive new information, 0 if filler.\n"
            "form: 1 when delivery is clean with coherent sentences and pacing. Target speed 12–18 characters per second; apply "
            "soft penalties at 10–12 or 18–20, strong penalty below 10 or above 20.\n"
            "importance: 1 if critical for the intent, decreasing as relevance drops.\n"
            "Return strict JSON: {\"segment_scores\": [{\"segment_id\": int, \"fact\": float, \"info\": float, \"form\": float, \"importance\": float, \"note\": str}]}"
        )
        user_prompt = (
            "Score the following segments. Each entry has segment_id, text, duration, and speed. "
            "Respond with JSON only.\n\n"
            + json.dumps(batch, ensure_ascii=False, indent=2)
        )

        response = call_openrouter(
            ctx,
            stage=f"scoring_batch_{batch_index:03d}",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            abort(f"Scoring stage returned invalid JSON: {exc}")

        if "segment_scores" not in data or not isinstance(data["segment_scores"], list):
            abort("Scoring stage output missing 'segment_scores'.")

        for entry in data["segment_scores"]:
            try:
                segment_id = int(entry["segment_id"])
                fact = float(entry["fact"])
                info = float(entry["info"])
                form = float(entry["form"])
                importance = float(entry["importance"])
                note = str(entry["note"])
            except (KeyError, TypeError, ValueError) as exc:
                abort(f"Invalid scoring entry: {exc}")

            stats_entry = next((item for item in batch if item["segment_id"] == segment_id), None)
            if stats_entry is None:
                abort(f"Scoring stage referenced unknown segment_id {segment_id}.")

            composite = (
                ctx.weights["fact"] * fact
                + ctx.weights["info"] * info
                + ctx.weights["form"] * form
                + ctx.weights["importance"] * importance
            )

            enriched = {
                "segment_id": segment_id,
                "duration": stats_entry["duration"],
                "speed": stats_entry["speed"],
                "scores": {
                    "fact": fact,
                    "info": info,
                    "form": form,
                    "importance": importance,
                    "note": note,
                },
                "composite_score": composite,
                "text": stats_entry["text"],
            }

            if fact >= ctx.args.fact_threshold and form >= ctx.args.form_threshold:
                kept.append(enriched)
            else:
                discarded.append(enriched)
                if fact < ctx.args.fact_threshold:
                    threshold_stats["fact"] += 1
                if form < ctx.args.form_threshold:
                    threshold_stats["form"] += 1

    payload = {"kept": kept, "discarded": discarded}
    dump_json(path, payload)
    append_log(
        ctx.plan_log,
        "Stage C complete – kept %d segments, discarded %d (fact below: %d, form below: %d)."
        % (len(kept), len(discarded), threshold_stats["fact"], threshold_stats["form"]),
    )

    if not kept:
        abort("All segments were discarded during scoring; no viable content remains.")

    return payload


def ensure_section_ids_contiguous(sections: List[Dict[str, Any]]) -> None:
    expected = 1
    for section in sections:
        sid = section.get("section_id")
        if sid != expected:
            abort("Section IDs must start at 1 and be contiguous.")
        expected += 1


def run_stage_d_outline(
    ctx: PipelineContext,
    intent_text: str,
    kept_segments: List[Dict[str, Any]],
    should_run: bool,
) -> List[Dict[str, Any]]:
    path = ctx.run_cache_dir / "outline.json"
    if not should_run and path.exists():
        outline = load_json(path)
        if not isinstance(outline, list):
            abort("outline.json must contain a list.")
        ensure_section_ids_contiguous(outline)
        return outline

    append_log(ctx.plan_log, "Running Stage D – Structure Planning")

    segment_samples = [
        {
            "text": seg["text"],
            "duration": seg["duration"],
            "speed": seg["speed"],
            "composite_score": seg["composite_score"],
        }
        for seg in kept_segments
    ]

    system_prompt = (
        "You are a senior video editor designing an outline for an edited cut. "
        "Use the intent summary and segment statistics to balance narrative flow, pacing, and engagement. "
        "Provide a JSON array of {section_id, title, goal, target_duration} with IDs starting at 1."
    )

    user_prompt = (
        f"Intent summary:\n{intent_text}\n\n"
        f"Target runtime: {ctx.target_length:.2f} seconds (0 means no limit).\n\n"
        "Segment samples (duration seconds, speed chars/sec, composite score 0-1):\n"
        f"{json.dumps(segment_samples, ensure_ascii=False, indent=2)}\n\n"
        "Design the outline now."
    )

    response = call_openrouter(
        ctx,
        stage="outline",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )

    try:
        outline = json.loads(response)
    except json.JSONDecodeError as exc:
        abort(f"Outline stage returned invalid JSON: {exc}")
    if not isinstance(outline, list):
        abort("Outline stage must return a JSON array.")
    ensure_section_ids_contiguous(outline)
    dump_json(path, outline)
    append_log(ctx.plan_log, "Stored outline.json")
    return outline


def run_stage_e_section_plan(
    ctx: PipelineContext,
    outline: List[Dict[str, Any]],
    kept_segments: List[Dict[str, Any]],
    should_run: bool,
) -> List[Dict[str, Any]]:
    path = ctx.run_cache_dir / "section_plan.json"
    if not should_run and path.exists():
        section_plan = load_json(path)
        if not isinstance(section_plan, list):
            abort("section_plan.json must contain a list.")
        return section_plan

    append_log(ctx.plan_log, "Running Stage E – Section Mapping")

    segments_payload = [
        {
            "segment_id": seg["segment_id"],
            "text": seg["text"],
            "duration": seg["duration"],
            "speed": seg["speed"],
            "composite_score": seg["composite_score"],
        }
        for seg in kept_segments
    ]

    cut_text = (
        "Every non-sequential jump between chosen segment IDs introduces a visible cut. "
        "Try to keep visible cuts ≤ %s per minute of final runtime. If tolerance is None, cuts are unrestricted."
        % ("None" if ctx.cut_tolerance is None else f"{ctx.cut_tolerance:.2f}")
    )

    system_prompt = (
        "You are a senior story editor assigning transcript segments to outline sections. "
        "Ensure each section meets its goal, respects pacing, and minimises cuts within the provided tolerance. "
        "Return a JSON array ordered like the outline, each entry {section_id, segment_ids, section_notes}."
    )
    user_prompt = (
        f"Outline:\n{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
        f"Segments:\n{json.dumps(segments_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Cut tolerance guidance: {cut_text}\n"
        f"Target runtime reminder: {ctx.target_length:.2f} seconds (0 means keep all viable content)."
    )

    response = call_openrouter(
        ctx,
        stage="section_plan",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )

    try:
        section_plan = json.loads(response)
    except json.JSONDecodeError as exc:
        abort(f"Section mapping stage returned invalid JSON: {exc}")
    if not isinstance(section_plan, list):
        abort("Section mapping stage must return a JSON array.")
    dump_json(path, section_plan)
    append_log(ctx.plan_log, "Stored section_plan.json")
    return section_plan


def flatten_section_plan(
    section_plan: List[Dict[str, Any]],
    kept_by_id: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sequence: List[Dict[str, Any]] = []
    for section in section_plan:
        try:
            section_id = int(section["section_id"])
            segment_ids = section["segment_ids"]
        except (KeyError, TypeError, ValueError) as exc:
            abort(f"Invalid section plan entry: {exc}")
        if not isinstance(segment_ids, list):
            abort("section_plan entries require 'segment_ids' list.")
        for seg_id in segment_ids:
            if seg_id not in kept_by_id:
                abort(f"section_plan references unknown segment_id {seg_id}.")
            seg = kept_by_id[seg_id]
            sequence.append(
                {
                    "segment_id": seg_id,
                    "section_id": section_id,
                    "text": seg["text"],
                    "duration": seg["duration"],
                    "speed": seg["speed"],
                    "composite_score": seg["composite_score"],
                }
            )
    return sequence


def compute_cuts(sequence: List[Dict[str, Any]]) -> Tuple[int, float]:
    if not sequence:
        return 0, 0.0
    cuts = 0
    for prev, curr in zip(sequence, sequence[1:]):
        if curr["segment_id"] != prev["segment_id"] + 1:
            cuts += 1
    total_duration = sum(item["duration"] for item in sequence)
    cuts_per_minute = cuts / (total_duration / 60.0) if total_duration > 0 else 0.0
    return cuts, cuts_per_minute


def run_stage_f_optimization(
    ctx: PipelineContext,
    intent_text: str,
    outline: List[Dict[str, Any]],
    section_plan: List[Dict[str, Any]],
    kept: List[Dict[str, Any]],
    should_run: bool,
) -> Dict[str, Any]:
    path = ctx.run_cache_dir / "optimization.json"
    if not should_run and path.exists():
        return load_json(path)

    kept_by_id = {int(item["segment_id"]): item for item in kept}
    sequence = flatten_section_plan(section_plan, kept_by_id)
    cuts, cuts_per_minute = compute_cuts(sequence)
    total_duration = sum(item["duration"] for item in sequence)

    tolerance_text = "None" if ctx.cut_tolerance is None else f"{ctx.cut_tolerance:.2f}"
    status_line = (
        f"Current plan duration: {total_duration:.2f}s. Visible cuts: {cuts} "
        f"({cuts_per_minute:.2f}/min). Target runtime: {ctx.target_length:.2f}s. "
        f"Cut tolerance: {tolerance_text}."
    )

    sequence_payload = [
        {
            "segment_id": item["segment_id"],
            "section_id": item["section_id"],
            "text": item["text"],
            "duration": item["duration"],
            "speed": item["speed"],
            "composite_score": item["composite_score"],
        }
        for item in sequence
    ]

    system_prompt = (
        "You are a finishing editor reviewing the section plan for a single-take recording. "
        "Treat the sequence like a final broadcast script. Focus on narrative clarity, "
        "emotional flow, and fulfillment of the stated intent. Use the numeric stats as "
        "sanity checks only. "
        "Return JSON {\"segment_ids\": [...], \"summary\": str} and ensure the segment_ids list is not empty."
    )

    cut_guidance = (
        "Every non-sequential jump introduces a visible cut. "
        "Try to keep cuts per minute within the provided tolerance unless narrative quality requires otherwise."
    )

    user_prompt = (
        f"Intent summary:\n{intent_text}\n\n"
        f"Outline:\n{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
        f"Section plan:\n{json.dumps(section_plan, ensure_ascii=False, indent=2)}\n\n"
        f"Status: {status_line}\n"
        f"Cut guidance: {cut_guidance}\n"
        f"Segments in play order:\n{json.dumps(sequence_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Target runtime: {ctx.target_length:.2f} seconds."
    )

    response = call_openrouter(
        ctx,
        stage="optimization",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )

    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        abort(f"Optimization stage returned invalid JSON: {exc}")

    if not isinstance(data, dict):
        abort("Optimization stage must return an object.")
    if "segment_ids" not in data or not isinstance(data["segment_ids"], list):
        abort("optimization.json must include a 'segment_ids' array.")
    if not data["segment_ids"]:
        abort("Optimization stage returned an empty segment list.")
    if "summary" not in data:
        abort("optimization.json must include a 'summary' field.")

    dump_json(path, data)
    append_log(ctx.plan_log, "Stored optimization.json")
    return data


def locate_segment(transcript: Dict[str, Any], segment_id: int) -> Dict[str, Any]:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        abort("transcript.json missing 'segments'.")
    for segment in segments:
        if int(segment.get("segment_id")) == segment_id:
            return segment
    abort(f"Segment ID {segment_id} not present in transcript.json.")


def get_file_path(transcript: Dict[str, Any], file_id: int) -> Path:
    files = transcript.get("files")
    if not isinstance(files, list):
        abort("transcript.json missing 'files'.")
    for entry in files:
        if int(entry.get("file_id")) == file_id:
            return Path(entry.get("path"))
    abort(f"file_id {file_id} not present in transcript.json.")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_ffmpeg_extract(
    ctx: PipelineContext,
    input_file: Path,
    start: float,
    end: float,
    output_file: Path,
) -> None:
    duration = max(end - start, 0.0)
    if duration <= 0:
        abort(f"Invalid segment with non-positive duration from {input_file}.")

    ensure_directory(output_file.parent)

    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_file),
    ]

    copy_cmd = base_cmd + ["-c", "copy", str(output_file)]
    code, _, stderr = run_subprocess(copy_cmd, ctx.render_log)
    if code != 0 or not output_file.exists() or output_file.stat().st_size < 1024:
        append_log(
            ctx.render_log,
            f"Stream copy failed or output tiny ({stderr.strip() if stderr else 'no stderr'}). Retrying with re-encode.",
        )
        reencode_cmd = base_cmd + [
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
            str(output_file),
        ]
        code, _, stderr = run_subprocess(reencode_cmd, ctx.render_log)
        if code != 0:
            abort(f"FFmpeg failed to extract segment from '{input_file}': {stderr}")


def run_ffmpeg_concat(
    ctx: PipelineContext,
    list_file: Path,
    output_file: Path,
) -> None:
    ensure_directory(output_file.parent)
    copy_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_file),
    ]
    code, _, stderr = run_subprocess(copy_cmd, ctx.render_log)
    if code != 0 or not output_file.exists() or output_file.stat().st_size < 4096:
        append_log(
            ctx.render_log,
            f"Concat stream copy failed or output tiny ({stderr.strip() if stderr else 'no stderr'}). Retrying with re-encode.",
        )
        reencode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
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
            str(output_file),
        ]
        code, _, stderr = run_subprocess(reencode_cmd, ctx.render_log)
        if code != 0:
            abort(f"FFmpeg failed to concat segments: {stderr}")


def run_stage_g_render(
    ctx: PipelineContext,
    transcript: Dict[str, Any],
    optimization: Dict[str, Any],
) -> Path:
    segment_ids = optimization.get("segment_ids")
    if not isinstance(segment_ids, list) or not segment_ids:
        abort("optimization.json contains no segments to render.")

    extract_dir = ctx.run_cache_dir / "render_tmp"
    ensure_directory(extract_dir)

    extracted_files: List[Path] = []
    for index, seg_id in enumerate(segment_ids):
        segment = locate_segment(transcript, int(seg_id))
        file_id = int(segment.get("file_id"))
        source_path = get_file_path(transcript, file_id)
        start = float(segment.get("start"))
        end = float(segment.get("end"))
        output = extract_dir / f"segment_{index:04d}.mp4"
        run_ffmpeg_extract(ctx, source_path, start, end, output)
        extracted_files.append(output)

    concat_list = extract_dir / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as handle:
        for file_path in extracted_files:
            handle.write(f"file '{file_path}'\n")

    final_output = ctx.run_output_dir / f"{ctx.stem}_cut.mp4"
    run_ffmpeg_concat(ctx, concat_list, final_output)
    append_log(ctx.render_log, f"Rendered final cut to '{final_output}'.")
    return final_output


def write_metadata(ctx: PipelineContext, metadata_payload: Dict[str, Any]) -> None:
    metadata = {"timestamp": _dt.datetime.now().isoformat(), "args": metadata_payload["args"]}
    dump_json(ctx.metadata_path, metadata)


def summarise_args(ctx: PipelineContext) -> Dict[str, Any]:
    return {
        "input_paths": [str(p) for p in ctx.input_paths],
        "language": ctx.args.language,
        "intent": ctx.args.intent,
        "cut_tolerance": ctx.cut_tolerance,
        "fact_threshold": ctx.args.fact_threshold,
        "form_threshold": ctx.args.form_threshold,
        "target_length": ctx.target_length,
        "score_weights": ctx.weights,
        "plan_only": ctx.args.plan_only,
        "whisper_model": ctx.args.whisper_model,
        "model": ctx.args.model,
    }


def orchestrate(ctx: PipelineContext) -> None:
    metadata_payload = {"args": summarise_args(ctx)}
    recompute_from, _ = determine_recompute_point(ctx, metadata_payload)

    ensure_dependencies(ctx, recompute_from)

    stage_should_run = lambda idx: recompute_from is None or idx >= recompute_from

    transcript = load_or_run_stage_a(ctx, stage_should_run(0))
    segments_stats = compute_segment_stats(transcript)
    intent_text = run_stage_b_intent(ctx, transcript, stage_should_run(1))
    scores = run_stage_c_scoring(ctx, segments_stats, stage_should_run(2))
    kept_segments = [
        {
            "segment_id": item["segment_id"],
            "text": item["text"],
            "duration": item["duration"],
            "speed": item["speed"],
            "composite_score": item["composite_score"],
        }
        for item in scores["kept"]
    ]
    outline = run_stage_d_outline(ctx, intent_text, kept_segments, stage_should_run(3))
    section_plan = run_stage_e_section_plan(ctx, outline, kept_segments, stage_should_run(4))
    optimization = run_stage_f_optimization(ctx, intent_text, outline, section_plan, scores["kept"], stage_should_run(5))

    write_metadata(ctx, metadata_payload)

    if ctx.args.plan_only:
        print(ctx.run_cache_dir / "optimization.json")
        return

    final_video = run_stage_g_render(ctx, transcript, optimization)
    print(final_video)


def build_context(args: argparse.Namespace, weights: Dict[str, float], target_length: float) -> PipelineContext:
    input_paths = expand_input_paths(args.input)
    stem = compute_stem(input_paths)
    cache_root = Path(args.cache_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    run_cache_dir = cache_root / stem
    run_output_dir = output_root / stem
    metadata_path = run_cache_dir / "metadata.json"
    plan_log = run_cache_dir / "plan.log"
    render_log = run_cache_dir / "render.log"
    debug_root = run_cache_dir / "debug"
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    return PipelineContext(
        args=args,
        input_paths=input_paths,
        stem=stem,
        cache_root=cache_root,
        run_cache_dir=run_cache_dir,
        output_root=output_root,
        run_output_dir=run_output_dir,
        metadata_path=metadata_path,
        plan_log=plan_log,
        render_log=render_log,
        debug_root=debug_root,
        weights=weights,
        target_length=target_length,
        cut_tolerance=args.cut_tolerance,
        api_key=None,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, weights, target_length = parse_args(argv)
    ctx = build_context(args, weights, target_length)
    try:
        orchestrate(ctx)
    except PipelineError as exc:
        append_log(ctx.plan_log, f"Fatal error: {exc}")
        raise


if __name__ == "__main__":
    main()

