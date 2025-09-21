#!/usr/bin/env python3
"""videocut4.py – deterministic single-take editing pipeline.

Implements the behaviour described in ``videocut4.md``.  The tool drives
Whisper transcription, GPT-assisted planning, and FFmpeg rendering while
respecting deterministic, cache-first execution.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("The 'requests' package is required to run videocut4.py") from exc


@dataclass(frozen=True)
class ScoreWeights:
    fact: float
    info: float
    form: float
    importance: float

    @classmethod
    def from_raw(cls, fact: float, info: float, form: float, importance: float) -> "ScoreWeights":
        weights = [fact, info, form, importance]
        total = sum(weights)
        if total <= 0:
            raise SystemExit("Score weights must sum to a positive value.")
        normalised = [value / total for value in weights]
        return cls(*normalised)


@dataclass
class CLIConfig:
    input_paths: List[Path]
    language: Optional[str]
    intent_text: Optional[str]
    cut_tolerance: Optional[float]
    fact_threshold: float
    form_threshold: float
    target_length: float
    weights: ScoreWeights
    cache_dir: Path
    output_dir: Path
    plan_only: bool
    debug: bool
    whisper_model: str
    openrouter_model: str


@dataclass
class StageContext:
    config: CLIConfig
    stem: str
    cache_dir: Path
    output_dir: Path
    metadata_path: Path
    plan_log: Path
    render_log: Path
    api_key: Optional[str]


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> CLIConfig:
    parser = argparse.ArgumentParser(description="Deterministic single-take video cutter")
    parser.add_argument("--input", nargs="+", required=True, help="Input media paths (supports glob patterns)")
    parser.add_argument("--language", choices=["en", "de"], default=None, help="Whisper language hint")
    parser.add_argument("--intent", default=None, help="Free-form user intent text")
    parser.add_argument("--cut-tolerance", type=float, default=None, help="Desired maximum cuts per minute")
    parser.add_argument("--fact-threshold", type=float, default=0.0, help="Minimum fact score to retain a segment")
    parser.add_argument("--form-threshold", type=float, default=0.3, help="Minimum form score to retain a segment")
    parser.add_argument("--target-length", default="0", help="Target runtime in seconds or mm:ss format")
    parser.add_argument("--score-weight-fact", type=float, default=0.25)
    parser.add_argument("--score-weight-info", type=float, default=0.25)
    parser.add_argument("--score-weight-form", type=float, default=0.25)
    parser.add_argument("--score-weight-importance", type=float, default=0.25)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--plan-only", action="store_true", help="Stop after generating optimization plan")
    parser.add_argument("--debug", action="store_true", help="Persist GPT prompts and responses")
    parser.add_argument("--whisper-model", default="small", help="Whisper model name")
    parser.add_argument("--model", default="openai/gpt-5", help="OpenRouter model identifier")

    namespace = parser.parse_args(argv)

    input_paths = expand_inputs(namespace.input)
    weights = ScoreWeights.from_raw(
        namespace.score_weight_fact,
        namespace.score_weight_info,
        namespace.score_weight_form,
        namespace.score_weight_importance,
    )
    target_length = parse_target_length(namespace.target_length)

    return CLIConfig(
        input_paths=input_paths,
        language=namespace.language,
        intent_text=namespace.intent,
        cut_tolerance=namespace.cut_tolerance,
        fact_threshold=namespace.fact_threshold,
        form_threshold=namespace.form_threshold,
        target_length=target_length,
        weights=weights,
        cache_dir=Path(namespace.cache_dir).resolve(),
        output_dir=Path(namespace.output_dir).resolve(),
        plan_only=namespace.plan_only,
        debug=namespace.debug,
        whisper_model=namespace.whisper_model,
        openrouter_model=namespace.model,
    )


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for pattern in patterns:
        matches = [Path(path).resolve() for path in sorted(glob.glob(pattern, recursive=True))]
        if not matches:
            raise SystemExit(f"Input pattern '{pattern}' did not match any files.")
        expanded.extend(matches)
    if not expanded:
        raise SystemExit("At least one input media path is required.")
    return expanded


def parse_target_length(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise SystemExit("Target length must be seconds or mm:ss format.")
        minutes, seconds = parts
        try:
            total = int(minutes) * 60 + float(seconds)
        except ValueError as exc:
            raise SystemExit("Invalid mm:ss target length provided.") from exc
        return max(total, 0.0)
    try:
        numeric = float(value)
    except ValueError as exc:
        raise SystemExit("Target length must be numeric seconds or mm:ss format.") from exc
    return max(numeric, 0.0)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe are required but were not found in PATH.")


def try_get_openrouter_api_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if os.name == "nt":  # pragma: no cover
        try:
            import winreg  # type: ignore

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as reg_key:
                value, _ = winreg.QueryValueEx(reg_key, "OPENROUTER_API_KEY")
                if value:
                    return str(value)
        except OSError:
            pass
    return None


def compute_stem(input_paths: Sequence[Path]) -> str:
    joined = "\n".join(str(path) for path in input_paths)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    return f"{input_paths[0].stem}_{digest}"


def snapshot_arguments(config: CLIConfig) -> Dict[str, Any]:
    return {
        "input_paths": [str(path) for path in config.input_paths],
        "language": config.language,
        "intent_text": config.intent_text,
        "cut_tolerance": config.cut_tolerance,
        "fact_threshold": config.fact_threshold,
        "form_threshold": config.form_threshold,
        "target_length": config.target_length,
        "score_weights": dataclasses.asdict(config.weights),
        "cache_dir": str(config.cache_dir),
        "output_dir": str(config.output_dir),
        "plan_only": config.plan_only,
        "debug": config.debug,
        "whisper_model": config.whisper_model,
        "openrouter_model": config.openrouter_model,
    }


def load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"metadata.json is corrupted: {exc}") from exc


def write_metadata(path: Path, snapshot: Dict[str, Any]) -> None:
    data = {"timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"), "args": snapshot}
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def log_message(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, debug_root: Path, debug_enabled: bool) -> None:
        self.api_key = api_key
        self.model = model
        self.debug_root = debug_root
        self.debug_enabled = debug_enabled

    def _debug_write(self, stage: str, payload: Dict[str, Any], response_text: str) -> None:
        if not self.debug_enabled:
            return
        stage_dir = self.debug_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        safe_payload = dict(payload)
        safe_payload.pop("headers", None)
        (stage_dir / "request.json").write_text(json.dumps(safe_payload, indent=2, sort_keys=True), encoding="utf-8")
        (stage_dir / "response.txt").write_text(response_text, encoding="utf-8")

    def call(self, messages: List[Dict[str, str]], *, temperature: float, stage: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response_text = response.text
        self._debug_write(stage, {"url": url, "json": payload}, response_text)
        if not response.ok:
            snippet = response_text[:4000]
            raise SystemExit(f"OpenRouter request failed ({response.status_code}): {snippet}")
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise SystemExit(f"OpenRouter returned invalid JSON: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit("Unexpected OpenRouter response schema.") from exc
        if not isinstance(content, str):
            raise SystemExit("OpenRouter response content was not a string.")
        return content


def find_existing_transcript_path(media_path: Path, cache_dir: Path) -> Optional[Path]:
    candidate = media_path.with_suffix(".json")
    if candidate.exists():
        return candidate
    cache_candidate = cache_dir / f"{media_path.stem}.json"
    if cache_candidate.exists():
        return cache_candidate
    return None


def read_transcript_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Transcript file '{path}' is invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or "segments" not in data:
        raise SystemExit(f"Transcript file '{path}' missing required keys.")
    segments = data["segments"]
    if not isinstance(segments, list) or not segments:
        raise SystemExit(f"Transcript file '{path}' must contain at least one segment.")
    for segment in segments:
        if not isinstance(segment, dict):
            raise SystemExit(f"Transcript file '{path}' contains invalid segment entries.")
        for key in ("start", "end", "text"):
            if key not in segment:
                raise SystemExit(f"Transcript segment in '{path}' missing key '{key}'.")
    return data


def sentence_from_words(words: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not words:
        return None
    text = "".join(word.get("word", "") for word in words).strip()
    if not text:
        return None
    start = words[0].get("start")
    end = words[-1].get("end")
    if start is None or end is None:
        return None
    return {"start": float(start), "end": float(end), "text": text}


def split_words_into_sentences(words: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sentences: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    for word in words:
        current.append(word)
        token = word.get("word", "")
        stripped = token.strip()
        if stripped and stripped[-1] in ".?!":
            sentence = sentence_from_words(current)
            if sentence:
                sentences.append(sentence)
            current = []
    if current:
        sentence = sentence_from_words(current)
        if sentence:
            sentences.append(sentence)
    return sentences


def run_whisper_transcription(media_path: Path, language: Optional[str], model_name: str) -> Dict[str, Any]:
    try:  # pragma: no cover
        import whisper  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("The 'whisper' package is required for transcription.") from exc

    model = whisper.load_model(model_name)
    options: Dict[str, Any] = {"word_timestamps": True}
    if language:
        options["language"] = language
    result = model.transcribe(str(media_path), **options)
    segments = result.get("segments") or []
    all_words: List[Dict[str, Any]] = []
    for segment in segments:
        words = segment.get("words") or []
        for word in words:
            if "start" in word and "end" in word:
                all_words.append(word)
    if not all_words:
        for segment in segments:
            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text")
            if start is not None and end is not None and isinstance(text, str):
                all_words.append({"start": start, "end": end, "word": text})
    sentences = split_words_into_sentences(all_words)
    if not sentences:
        raise SystemExit(f"Whisper produced no usable transcript for '{media_path}'.")
    return {"filename": media_path.name, "segments": sentences}


def ensure_transcripts(config: CLIConfig, context: StageContext) -> Dict[str, Any]:
    transcripts: List[Dict[str, Any]] = []
    for media_path in config.input_paths:
        existing = find_existing_transcript_path(media_path, context.cache_dir)
        if existing is not None:
            data = read_transcript_file(existing)
        else:
            log_message(context.plan_log, f"Transcribing {media_path.name} with Whisper model '{config.whisper_model}'")
            data = run_whisper_transcription(media_path, config.language, config.whisper_model)
            target_path = media_path.with_suffix(".json")
            write_path = target_path
            try:
                target_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            except OSError:
                log_message(context.plan_log, f"Falling back to cache directory for transcript of {media_path}")
                write_path = context.cache_dir / f"{media_path.stem}.json"
                data["filename"] = str(media_path.resolve())
                write_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        transcripts.append(data)

    files_table: List[Dict[str, Any]] = []
    segments_out: List[Dict[str, Any]] = []
    segment_id = 0
    for file_id, (media_path, transcript) in enumerate(zip(config.input_paths, transcripts)):
        files_table.append({"file_id": file_id, "path": str(media_path)})
        for segment in transcript["segments"]:
            start = float(segment["start"])
            end = float(segment["end"])
            text = str(segment["text"])
            segments_out.append(
                {
                    "segment_id": segment_id,
                    "file_id": file_id,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
            segment_id += 1

    if not segments_out:
        raise SystemExit("No transcript segments were produced.")

    transcript_payload = {"files": files_table, "segments": segments_out}
    transcript_path = context.cache_dir / "transcript.json"
    transcript_path.write_text(json.dumps(transcript_payload, indent=2, sort_keys=True), encoding="utf-8")
    log_message(context.plan_log, f"Wrote merged transcript with {len(segments_out)} segments.")
    return transcript_payload


def load_transcript(context: StageContext) -> Dict[str, Any]:
    path = context.cache_dir / "transcript.json"
    if not path.exists():
        raise SystemExit("transcript.json is missing; rerun Stage A.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"transcript.json is invalid: {exc}") from exc
    return data


def build_intent_prompt(transcript: Dict[str, Any], intent_text: Optional[str]) -> str:
    segments = transcript.get("segments", [])
    lines = [seg["text"] for seg in segments]
    transcript_text = "\n".join(lines)
    intent_section = intent_text.strip() if intent_text else "(No explicit user intent provided.)"
    return (
        "You will receive the transcript of one or more single-take recordings. Summarise the video's purpose, audience, and tone in 2-3 bullets, "
        "then organise key themes into four buckets: **Essential**, **Important**, **Optional**, **Leave out**. Keep the response under 250 words and do not reference segment IDs or timings.\n\n"
        "Transcript:\n" + transcript_text + "\n\nUser intent hint:\n" + intent_section
    )


def run_intent_stage(context: StageContext, transcript: Dict[str, Any]) -> str:
    if context.api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is required for the intent stage.")
    client = OpenRouterClient(context.api_key, context.config.openrouter_model, context.cache_dir / "debug", context.config.debug)
    prompt = build_intent_prompt(transcript, context.config.intent_text)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an editorial strategist distilling raw transcripts into concise planning notes. Summaries must be precise, actionable, and within the requested word limit."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    response = client.call(messages, temperature=0.0, stage="intent")
    intent_text = response.rstrip()
    (context.cache_dir / "intent.md").write_text(intent_text + "\n", encoding="utf-8")
    log_message(context.plan_log, "Intent stage completed.")
    return intent_text


def load_intent(context: StageContext) -> str:
    path = context.cache_dir / "intent.md"
    if not path.exists():
        raise SystemExit("intent.md is missing; rerun Stage B.")
    return path.read_text(encoding="utf-8").rstrip("\n")


def compute_segment_metrics(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = transcript.get("segments", [])
    metrics: List[Dict[str, Any]] = []
    for segment in segments:
        duration = float(segment["end"]) - float(segment["start"])
        duration = max(duration, 0.0)
        text = str(segment["text"])
        speed = len(text) / max(duration, 1e-6)
        metrics.append(
            {
                "segment_id": int(segment["segment_id"]),
                "file_id": int(segment["file_id"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
                "duration": duration,
                "speed": speed,
            }
        )
    return metrics


def run_scoring_stage(context: StageContext, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if context.api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is required for the scoring stage.")
    client = OpenRouterClient(context.api_key, context.config.openrouter_model, context.cache_dir / "debug", context.config.debug)

    batches: List[List[Dict[str, Any]]] = []
    batch_size = 50
    for index in range(0, len(segments), batch_size):
        batches.append(segments[index : index + batch_size])

    kept_entries: List[Dict[str, Any]] = []
    discarded_entries: List[Dict[str, Any]] = []

    for batch_index, batch in enumerate(batches):
        payload = [
            {
                "segment_id": entry["segment_id"],
                "text": entry["text"],
                "duration": entry["duration"],
                "speed": entry["speed"],
            }
            for entry in batch
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You score transcript segments for factual accuracy, informational value, form, and importance. "
                    "fact: +1 truthful/precise, 0 neutral, negative when incorrect. "
                    "info: 1 when the segment adds substantive information, 0 when filler. "
                    "form: 1 when delivery is clean with coherent sentences, minimal filler, and pacing informed by the provided speed. Target pacing is roughly 12-18 characters/sec (12-16 acceptable for dense or slower deliveries); apply a soft penalty at 10-12 or 18-20 and a strong penalty below 10 or above 20. "
                    "importance: 1 when critical to fulfilling the intent. Provide a short note justifying the scores. Respond strictly as JSON matching {'segment_scores': [...]}"
                ),
            },
            {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=False)},
        ]
        response = client.call(messages, temperature=0.0, stage=f"scoring_batch_{batch_index:03d}")
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Scoring response could not be parsed: {exc}") from exc
        segment_scores = parsed.get("segment_scores")
        if not isinstance(segment_scores, list):
            raise SystemExit("Scoring response missing 'segment_scores'.")
        for entry in segment_scores:
            try:
                segment_id = int(entry["segment_id"])
                fact = float(entry["fact"])
                info = float(entry["info"])
                form = float(entry["form"])
                importance = float(entry["importance"])
                note = str(entry.get("note", ""))
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit("Scoring response entry missing required fields.") from exc
            source = next((item for item in segments if item["segment_id"] == segment_id), None)
            if source is None:
                raise SystemExit(f"Scoring response referenced unknown segment {segment_id}.")
            composite = (
                context.config.weights.fact * fact
                + context.config.weights.info * info
                + context.config.weights.form * form
                + context.config.weights.importance * importance
            )
            scored_entry = {
                "segment_id": segment_id,
                "duration": source["duration"],
                "speed": source["speed"],
                "scores": {
                    "fact": fact,
                    "info": info,
                    "form": form,
                    "importance": importance,
                    "note": note,
                },
                "composite_score": composite,
                "text": source["text"],
            }
            if fact >= context.config.fact_threshold and form >= context.config.form_threshold:
                kept_entries.append(scored_entry)
            else:
                discarded_entries.append(scored_entry)

    log_message(
        context.plan_log,
        f"Scoring complete: kept {len(kept_entries)} segments, discarded {len(discarded_entries)} below thresholds.",
    )
    output = {"kept": kept_entries, "discarded": discarded_entries}
    (context.cache_dir / "scores.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_scores(context: StageContext) -> Dict[str, Any]:
    path = context.cache_dir / "scores.json"
    if not path.exists():
        raise SystemExit("scores.json missing; rerun Stage C.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"scores.json is invalid: {exc}") from exc


def build_outline_prompt(intent: str, kept_segments: List[Dict[str, Any]], target_length: float) -> str:
    segment_samples = [
        {
            "text": entry["text"],
            "duration": entry["duration"],
            "speed": entry["speed"],
            "composite_score": entry["composite_score"],
        }
        for entry in kept_segments
    ]
    payload = {
        "intent": intent,
        "target_runtime_seconds": target_length,
        "segments": segment_samples,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_outline_stage(context: StageContext, intent: str, kept_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if context.api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is required for structure planning.")
    client = OpenRouterClient(context.api_key, context.config.openrouter_model, context.cache_dir / "debug", context.config.debug)
    prompt = build_outline_prompt(intent, kept_segments, context.config.target_length)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior producer designing a narrative outline for an edited video. Create contiguous sections that take the viewer from introduction to core material to a wrap-up, balancing flow and engagement. Output must be JSON array objects with section_id, title, goal, target_duration. IDs start at 1 and are contiguous."
            ),
        },
        {
            "role": "user",
            "content": (
                "Design an outline for the following intent, target runtime, and segment statistics. Respect the JSON output format.\n\n"
                + prompt
            ),
        },
    ]
    response = client.call(messages, temperature=0.0, stage="outline")
    try:
        outline = json.loads(response)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Outline response invalid JSON: {exc}") from exc
    if not isinstance(outline, list) or not outline:
        raise SystemExit("Outline response must be a non-empty JSON array.")
    for index, section in enumerate(outline, start=1):
        if not isinstance(section, dict):
            raise SystemExit("Outline entries must be objects.")
        if int(section.get("section_id", index)) != index:
            raise SystemExit("Outline section IDs must be contiguous starting at 1.")
    (context.cache_dir / "outline.json").write_text(json.dumps(outline, indent=2, sort_keys=True), encoding="utf-8")
    log_message(context.plan_log, f"Outline stage produced {len(outline)} sections.")
    return outline


def load_outline(context: StageContext) -> List[Dict[str, Any]]:
    path = context.cache_dir / "outline.json"
    if not path.exists():
        raise SystemExit("outline.json missing; rerun Stage D.")
    try:
        outline = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"outline.json invalid: {exc}") from exc
    if not isinstance(outline, list):
        raise SystemExit("outline.json must contain a JSON array.")
    return outline


def build_section_mapping_prompt(
    outline: List[Dict[str, Any]],
    kept_segments: List[Dict[str, Any]],
    cut_tolerance: Optional[float],
    target_length: float,
) -> str:
    if cut_tolerance is not None:
        tolerance_text = (
            "Every non-sequential jump between chosen segment IDs introduces a visible cut. "
            f"Try to keep visible cuts ≤ {cut_tolerance:.2f} per minute of final runtime."
        )
    else:
        tolerance_text = "Visible cuts are unrestricted."
    payload = {
        "outline": outline,
        "segments": [
            {
                "segment_id": entry["segment_id"],
                "text": entry["text"],
                "duration": entry["duration"],
                "speed": entry["speed"],
                "composite_score": entry["composite_score"],
            }
            for entry in kept_segments
        ],
        "cut_tolerance": tolerance_text,
        "target_runtime_seconds": target_length,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_section_mapping_stage(
    context: StageContext,
    outline: List[Dict[str, Any]],
    kept_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if context.api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is required for section mapping.")
    client = OpenRouterClient(context.api_key, context.config.openrouter_model, context.cache_dir / "debug", context.config.debug)
    prompt = build_section_mapping_prompt(
        outline,
        kept_segments,
        context.config.cut_tolerance,
        context.config.target_length,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a video editor assigning transcript segments to outline sections. For each section choose the strongest available segments to satisfy the goal while honouring flow. Prefer natural adjacency when it supports the narrative, but reach for later segments when necessary. Remember that every non-sequential jump adds a visible cut; keep cuts within the provided tolerance when possible. Include section_notes summarising trade-offs. Output must be JSON array entries {section_id, segment_ids, section_notes}."
            ),
        },
        {
            "role": "user",
            "content": "Plan segment assignments for the outline and segments below.\n\n" + prompt,
        },
    ]
    response = client.call(messages, temperature=0.0, stage="section_plan")
    try:
        section_plan = json.loads(response)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Section mapping response invalid JSON: {exc}") from exc
    if not isinstance(section_plan, list) or not section_plan:
        raise SystemExit("Section mapping must return a non-empty JSON array.")
    (context.cache_dir / "section_plan.json").write_text(
        json.dumps(section_plan, indent=2, sort_keys=True), encoding="utf-8"
    )
    log_message(context.plan_log, "Section mapping stage completed.")
    return section_plan


def load_section_plan(context: StageContext) -> List[Dict[str, Any]]:
    path = context.cache_dir / "section_plan.json"
    if not path.exists():
        raise SystemExit("section_plan.json missing; rerun Stage E.")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"section_plan.json invalid: {exc}") from exc
    if not isinstance(plan, list):
        raise SystemExit("section_plan.json must contain a JSON array.")
    return plan


def load_optimization(context: StageContext) -> Dict[str, Any]:
    path = context.cache_dir / "optimization.json"
    if not path.exists():
        raise SystemExit("optimization.json missing; rerun Stage F.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"optimization.json invalid: {exc}") from exc
    return data


def cut_tolerance_text(value: Optional[float]) -> str:
    if value is None:
        return "Visible cuts are unrestricted; focus on narrative needs."
    return (
        "Every non-sequential jump between chosen segment IDs introduces a visible cut. "
        f"Aim to keep visible cuts ≤ {value:.2f} per minute of final runtime."
    )


def format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:d}m{remainder:05.2f}s"


def flatten_section_plan(plan: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    sequence: List[Tuple[int, int]] = []
    for entry in plan:
        if not isinstance(entry, dict):
            raise SystemExit("section_plan entries must be objects.")
        section_id = int(entry.get("section_id"))
        segment_ids = entry.get("segment_ids")
        if not isinstance(segment_ids, list):
            raise SystemExit("section_plan entries require a list of segment_ids.")
        for seg_id in segment_ids:
            sequence.append((section_id, int(seg_id)))
    return sequence


def compute_cut_statistics(sequence: List[int], segments_map: Dict[int, Dict[str, Any]]) -> Tuple[float, int, float]:
    total_duration = sum(segments_map[seg_id]["duration"] for seg_id in sequence if seg_id in segments_map)
    visible_cuts = 0
    for prev, curr in zip(sequence, sequence[1:]):
        if curr != prev + 1:
            visible_cuts += 1
    cuts_per_minute = 0.0
    if total_duration > 0:
        cuts_per_minute = visible_cuts / (total_duration / 60.0)
    return total_duration, visible_cuts, cuts_per_minute


def build_optimization_context(
    intent: str,
    outline: List[Dict[str, Any]],
    section_plan: List[Dict[str, Any]],
    kept_segments: List[Dict[str, Any]],
    cut_tolerance: Optional[float],
    target_length: float,
) -> Tuple[str, List[Dict[str, Any]], str]:
    sequence_pairs = flatten_section_plan(section_plan)
    segment_map = {entry["segment_id"]: entry for entry in kept_segments}
    ordered_segment_ids = [seg_id for _, seg_id in sequence_pairs]
    total_duration, visible_cuts, cuts_per_minute = compute_cut_statistics(ordered_segment_ids, segment_map)
    status_line = (
        f"Current runtime {format_seconds(total_duration)} ({total_duration:.2f}s); "
        f"visible cuts {visible_cuts} ({cuts_per_minute:.2f}/min); "
        f"target runtime {'no limit' if target_length == 0 else f'{target_length:.2f}s'}; "
        f"cut tolerance: {cut_tolerance_text(cut_tolerance)}"
    )
    playback_list = []
    for section_id, segment_id in sequence_pairs:
        if segment_id not in segment_map:
            raise SystemExit(f"Section plan referenced unknown segment {segment_id}.")
        data = segment_map[segment_id]
        playback_list.append(
            {
                "segment_id": segment_id,
                "section_id": section_id,
                "text": data["text"],
                "duration": data["duration"],
                "speed": data["speed"],
                "composite_score": data["composite_score"],
            }
        )
    context_payload = {
        "intent": intent,
        "outline": outline,
        "section_plan": section_plan,
        "target_runtime_seconds": target_length,
        "cut_tolerance_note": cut_tolerance_text(cut_tolerance),
        "status": status_line,
        "playback_segments": playback_list,
    }
    return json.dumps(context_payload, indent=2, ensure_ascii=False), playback_list, status_line


def run_optimization_stage(
    context: StageContext,
    intent: str,
    outline: List[Dict[str, Any]],
    section_plan: List[Dict[str, Any]],
    kept_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if context.api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is required for optimization.")
    client = OpenRouterClient(context.api_key, context.config.openrouter_model, context.cache_dir / "debug", context.config.debug)
    payload_text, _playback_list, status_line = build_optimization_context(
        intent,
        outline,
        section_plan,
        kept_segments,
        context.config.cut_tolerance,
        context.config.target_length,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an editorial director finalising a broadcast-ready cut. Treat the plan as a script: read for clarity, emotional cadence, and fulfilment of the stated intent. Use pacing statistics as guardrails, not hard targets. Respond with JSON containing 'segment_ids' (final order) and 'summary' describing the key editorial choices."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review the current plan, status, and segments. Adjust the order to maximise narrative impact while respecting the guidance. Return JSON with final segment_ids and a concise summary.\n\n"
                + payload_text
            ),
        },
    ]
    response = client.call(messages, temperature=0.0, stage="optimization")
    try:
        optimisation = json.loads(response)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Optimization response invalid JSON: {exc}") from exc
    if not isinstance(optimisation, dict):
        raise SystemExit("Optimization response must be a JSON object.")
    segment_ids = optimisation.get("segment_ids")
    if not isinstance(segment_ids, list) or not segment_ids:
        raise SystemExit("Optimization must provide a non-empty list of segment_ids.")
    if not isinstance(optimisation.get("summary"), str):
        raise SystemExit("Optimization summary must be a string.")
    (context.cache_dir / "optimization.json").write_text(
        json.dumps(optimisation, indent=2, sort_keys=True), encoding="utf-8"
    )
    log_message(context.plan_log, f"Optimization completed. Status before optimisation: {status_line}")
    return optimisation


def build_render_segments(
    optimization: Dict[str, Any],
    transcript: Dict[str, Any],
    scores: Dict[str, Any],
) -> List[Dict[str, Any]]:
    segment_ids = optimization.get("segment_ids")
    if not isinstance(segment_ids, list) or not segment_ids:
        raise SystemExit("optimization.json contains no segment_ids.")
    transcript_segments = {int(seg["segment_id"]): seg for seg in transcript.get("segments", [])}
    if not transcript_segments:
        raise SystemExit("transcript.json has no segments.")
    kept_map = {int(entry["segment_id"]): entry for entry in scores.get("kept", [])}
    render_segments: List[Dict[str, Any]] = []
    for seg_id in segment_ids:
        if seg_id not in transcript_segments:
            raise SystemExit(f"Optimization referenced unknown segment {seg_id}.")
        segment = transcript_segments[seg_id]
        score_entry = kept_map.get(seg_id)
        render_segments.append(
            {
                "segment_id": seg_id,
                "file_id": int(segment["file_id"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment.get("text", ""),
                "scores": score_entry,
            }
        )
    return render_segments


def ffmpeg_time(value: float) -> str:
    return f"{value:.3f}"


def run_command(command: List[str], log_path: Path) -> None:
    log_message(log_path, "Running: " + " ".join(command))
    process = subprocess.run(command, capture_output=True, text=True)
    output = (process.stdout or "") + (process.stderr or "")
    if output:
        for line in output.splitlines():
            log_message(log_path, line)
    if process.returncode != 0:
        raise SystemExit(f"Command failed ({process.returncode}): {' '.join(command)}")


def extract_segment(
    source: Path,
    start: float,
    end: float,
    destination: Path,
    log_path: Path,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        ffmpeg_time(start),
        "-to",
        ffmpeg_time(end),
        "-i",
        str(source),
        "-c",
        "copy",
        str(destination),
    ]
    try:
        run_command(command, log_path)
        if destination.stat().st_size < 1024:
            destination.unlink(missing_ok=True)
            raise SystemExit("segment too small")
    except SystemExit:
        reencode = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            ffmpeg_time(start),
            "-to",
            ffmpeg_time(end),
            "-i",
            str(source),
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
            str(destination),
        ]
        run_command(reencode, log_path)


def concat_segments(segment_files: List[Path], output_path: Path, log_path: Path) -> None:
    concat_file = output_path.with_suffix(".txt")
    concat_file.write_text("\n".join(f"file '{path}'" for path in segment_files) + "\n", encoding="utf-8")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        run_command(command, log_path)
        if output_path.stat().st_size < 1024:
            output_path.unlink(missing_ok=True)
            raise SystemExit("concat too small")
    except SystemExit:
        fallback = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
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
            str(output_path),
        ]
        run_command(fallback, log_path)
    finally:
        concat_file.unlink(missing_ok=True)


def render_video(
    context: StageContext,
    optimization: Dict[str, Any],
    transcript: Dict[str, Any],
    scores: Dict[str, Any],
) -> Path:
    render_segments = build_render_segments(optimization, transcript, scores)
    if not render_segments:
        raise SystemExit("Optimization produced zero segments; cannot render an empty slate.")

    files_table = {int(file_entry["file_id"]): file_entry["path"] for file_entry in transcript.get("files", [])}
    if not files_table:
        raise SystemExit("transcript.json missing file mappings.")

    context.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = context.output_dir / f"{context.stem}_cut.mp4"

    with tempfile.TemporaryDirectory(dir=context.cache_dir) as tmpdir:
        temp_paths: List[Path] = []
        for index, segment in enumerate(render_segments):
            source_path = Path(files_table[int(segment["file_id"])])
            destination = Path(tmpdir) / f"segment_{index:04d}.mp4"
            extract_segment(source_path, segment["start"], segment["end"], destination, context.render_log)
            temp_paths.append(destination)
        concat_segments(temp_paths, final_path, context.render_log)

    if final_path.stat().st_size < 1024:
        raise SystemExit("Rendered video is suspiciously small after concat attempts.")

    print(str(final_path))
    log_message(context.render_log, f"Rendering complete: {final_path}")
    return final_path


def stage_outputs_present(context: StageContext, stage_index: int) -> bool:
    if stage_index == 0:
        if not (context.cache_dir / "transcript.json").exists():
            return False
        for media_path in context.config.input_paths:
            if find_existing_transcript_path(media_path, context.cache_dir) is None:
                return False
        return True
    if stage_index == 1:
        return (context.cache_dir / "intent.md").exists()
    if stage_index == 2:
        return (context.cache_dir / "scores.json").exists()
    if stage_index == 3:
        return (context.cache_dir / "outline.json").exists()
    if stage_index == 4:
        return (context.cache_dir / "section_plan.json").exists()
    if stage_index == 5:
        return (context.cache_dir / "optimization.json").exists()
    return False


def run_pipeline(config: CLIConfig) -> None:
    ensure_ffmpeg_available()
    stem = compute_stem(config.input_paths)
    cache_dir = config.cache_dir / stem
    output_dir = config.output_dir / stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = cache_dir / "metadata.json"
    plan_log = cache_dir / "plan.log"
    render_log = cache_dir / "render.log"

    snapshot = snapshot_arguments(config)
    previous = load_metadata(metadata_path)

    args_match = previous is not None and previous.get("args") == snapshot

    start_stage = 0
    if args_match:
        start_stage = 6
        for index in range(6):
            if not stage_outputs_present(StageContext(config, stem, cache_dir, output_dir, metadata_path, plan_log, render_log, None), index):
                start_stage = index
                break
    else:
        start_stage = 0

    context = StageContext(config, stem, cache_dir, output_dir, metadata_path, plan_log, render_log, None)

    needs_gpt = any(stage >= 1 for stage in range(start_stage, 6))
    api_key = try_get_openrouter_api_key() if needs_gpt else None
    if needs_gpt and not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for the requested stages.")
    context.api_key = api_key

    transcript = None
    intent = None
    scores = None
    outline = None
    section_plan = None
    optimization = None

    # Stage A
    if start_stage <= 0:
        log_message(plan_log, "Starting Stage A – Transcript Harvest")
        transcript = ensure_transcripts(config, context)
    else:
        transcript = load_transcript(context)

    # Stage B
    if start_stage <= 1:
        log_message(plan_log, "Starting Stage B – Intent")
        intent = run_intent_stage(context, transcript)
    else:
        intent = load_intent(context)

    segment_metrics = compute_segment_metrics(transcript)

    # Stage C
    if start_stage <= 2:
        log_message(plan_log, "Starting Stage C – Scoring")
        scores = run_scoring_stage(context, segment_metrics)
    else:
        scores = load_scores(context)

    kept_segments = scores.get("kept", [])
    if not kept_segments:
        raise SystemExit("All segments were discarded after scoring; cannot continue.")

    # Stage D
    if start_stage <= 3:
        log_message(plan_log, "Starting Stage D – Structure Planning")
        outline = run_outline_stage(context, intent, kept_segments)
    else:
        outline = load_outline(context)

    # Stage E
    if start_stage <= 4:
        log_message(plan_log, "Starting Stage E – Section Mapping")
        section_plan = run_section_mapping_stage(context, outline, kept_segments)
    else:
        section_plan = load_section_plan(context)

    # Stage F
    if start_stage <= 5:
        log_message(plan_log, "Starting Stage F – Optimization")
        optimization = run_optimization_stage(context, intent, outline, section_plan, kept_segments)
    else:
        optimization = load_optimization(context)

    write_metadata(metadata_path, snapshot)

    if config.plan_only:
        print(str(context.cache_dir / "optimization.json"))
        return

    render_video(context, optimization, transcript, scores)


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    main()

