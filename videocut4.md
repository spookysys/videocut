**videocut4.py – Detailed Specification**

Build a Python 3.10+ CLI that ingests one or more single-take recordings, transcribes them with Whisper, plans an edited cut via GPT (OpenRouter), and renders the result with FFmpeg. The tool must be deterministic, cache-first, and fail fast on missing dependencies, invalid data, or unmet constraints. Use only the behaviour described below; assume no prior versions.

---

### 1. CLI Contract

```
videocut4.py --input <paths...> [options]

Required:
  --input            One or more media paths. Accept glob patterns (expand in CLI order).

Optional:
  --language         Whisper hint: "en", "de", or omit to auto-detect.
  --intent           Free-form text the user supplies about the video’s intent; pass verbatim to the intent stage.
  --cut-tolerance    Float (cuts per minute) describing desired max visible cuts; None/omitted means no limit.
  --fact-threshold   Float [-1, 1], default 0.0. Segments below are discarded.
  --form-threshold   Float [0, 1], default 0.3. Segments below are discarded.
  --target-length    Seconds or mm:ss string; default 0 (keep all viable content).
  --score-weight-fact        Float [0,1], default 0.25.
  --score-weight-info        Float [0,1], default 0.25.
  --score-weight-form        Float [0,1], default 0.25.
  --score-weight-importance  Float [0,1], default 0.25.
  --cache-dir        Root for caches (default ./cache).
  --output-dir       Root for rendered output (default ./output).
  --plan-only        Stop after optimization; skip rendering.
  --debug            Persist prompt/response payloads for GPT calls (scrub secrets).
  --whisper-model    Whisper model name (default small).
  --model            OpenRouter model ID (default openai/gpt-5).
```

Normalize the four score weights' sum to 1. Abort immediately if ffmpeg/ffprobe are missing or, when recomputation needs GPT, `OPENROUTER_API_KEY` is unavailable (also try Windows registry keys `Environment`).

---

### 2. Cache Layout

For each run, derive a deterministic stem:

- Expand `--input` paths (globs) to absolute paths in the order provided.
- Stem = `<basename_of_first_input>_<8-char_sha1_of_joined_paths>`.
- Cache lives in `<cache-dir>/<stem>/`, output in `<output-dir>/<stem>/`.

Artefacts:

1. `metadata.json` – captured CLI args (resolved paths) and timestamp.
2. Per input file transcripts: `<media_basename>.json` next to the media file (if writable) else in the cache root. Schema:
   ```json
   { "filename": "videoA.mp4", "segments": [ { "start": 12.14, "end": 18.87, "text": "..." }, ... ] }
   ```
   When stored in cache because the source folder is read-only, use the absolute path in `"filename"`.
3. `transcript.json` – merged transcript for all inputs:
   ```json
   {
     "files": [ { "file_id": 0, "path": "videoA.mp4" }, ... ],
     "segments": [
       { "segment_id": 0, "file_id": 0, "start": 12.14, "end": 18.87, "text": "..." },
       { "segment_id": 1, "file_id": 0, "start": 18.87, "end": 27.41, "text": "..." },
       { "segment_id": 2, "file_id": 1, "start": 4.12, "end": 9.87,  "text": "..." }
     ]
   }
   ```
4. `intent.md`
5. `scores.json`
6. `outline.json`
7. `section_plan.json`
8. `optimization.json`
9. `plan.log` and `render.log`
10. `debug/<stage>/request.json` + `response.txt` when `--debug`.

Recompute cascade: deleting an artefact forces that stage and all downstream stages to rerun.

---

### 3. Stage-by-Stage Behaviour

#### Stage A – Transcript Harvest (no GPT)

1. For each input media file:
   - Look for `<basename>.json` in the same directory; if present, use it.
   - If absent: run Whisper (`word_timestamps=True`, language hint when provided). Use the word-level timings to re-segment the transcript into full sentences (deterministic splitter, e.g. `.?!` boundaries) and, for each sentence, set `start = first word start`, `end = last word end`.
   - Write the sentence-level transcript JSON next to the media file, or to the cache directory when the media path is unwritable.
2. Build `transcript.json`:
   - Assign sequential `segment_id` values in CLI order across all files.
   - Include `file_id` references based on the `files` table.
   - Store `start`, `end`, and `text` only.

If any transcript generation fails, abort. Treat missing segments as fatal.

#### Stage B – Intent (GPT)

- Construct transcript text by concatenating `transcript.json["segments"]` in order. Append the CLI `--intent` text as an explicit “User intent hint” section inside the prompt.
- Prompt: summarise purpose/audience/tone; produce prioritized content buckets (Essential / Important / Optional / Leave out). Stress conciseness.
- Save raw response as `intent.md` (strip trailing whitespace only). This stage runs whenever `intent.md` is missing or inputs changed.

#### Stage C – Scoring (GPT + post)

1. Precompute for each segment:
   - `duration = end - start`
   - `speed = len(text) / max(duration, 1e-6)`
2. Batch segments (≈50 per call). Prompt each batch with JSON array entries `{ "segment_id": int, "text": str, "duration": float, "speed": float }`.
   - System instructions define scoring rubric for `fact`, `info`, `form`, `importance`, and request a `note`.
   - No composite score requested.
3. After each response:
   - Parse strict JSON (`{"segment_scores":[...]}`).
   - Add `duration`, `speed`, `composite_score = wf*fact + wi*info + wform*form + wimp*importance` using CLI weights.
   - Store scores inside a nested dict:
     ```json
     "scores": {"fact":..., "info":..., "form":..., "importance":..., "note":"..."}
     ```
4. Partition segments:
   - `kept` where `fact >= fact_threshold` **and** `form >= form_threshold`.
   - `discarded` for everything else.
5. Write `scores.json`:

   ```json
   {
     "kept": [
       {
         "segment_id": 12,
         "duration": 8.53,
         "speed": 17.2,
         "scores": { "fact": 0.8, "info": 0.9, "form": 0.7, "importance": 0.9, "note": "…" },
         "composite_score": 0.83
       }
     ],
     "discarded": [ ... same structure ... ]
   }
   ```

No segment below thresholds may appear in later stages. Log how many were discarded and why.

#### Stage D – Structure Planning (GPT)

- Inputs: `intent.md`, target runtime, and the kept segments reduced to `{ "text": str, "duration": float, "speed": float, "composite_score": float }`.
- Prompt the model to design an ordered outline tailored to the intent, referencing target duration and pacing guidance implied by duration/speed/composite score stats.
- Output `outline.json`:

  ```json
  [
    { "section_id": 1, "title": "...", "goal": "...", "target_duration": 90.0 },
    { "section_id": 2, ... }
  ]
  ```

Ensure section IDs start at 1 and are contiguous. Allow user edits later; they must be preserved if present.

#### Stage E – Section Mapping (GPT)

- Gather context:
  - The outline (`section_id`, `title`, `goal`, `target_duration`).
  - Complete kept segment list re-joined with IDs: `{ "segment_id": int, "text": str, "duration": float, "speed": float, "composite_score": float }`.
  - Numeric cut tolerance plus rubric text: “Every non-sequential jump between chosen segment IDs introduces a visible cut. Try to keep visible cuts ≤ `cut_tolerance` per minute of final runtime. If tolerance is `None`, cuts are unrestricted.”
  - Target runtime reminder.
- Prompt instructions: For each section, select and order segment IDs to satisfy the goal, minimise visible cuts given the tolerance, and note compromises or difficult choices in `section_notes`.
- Output `section_plan.json`:

  ```json
  [
    { "section_id": 1, "segment_ids": [5, 6, 9], "section_notes": "One jump (6→9) to cover diagnosis example." },
    ...
  ]
  ```

#### Stage F – Optimization (GPT)

- Inputs:
  - `intent.md`
  - `outline.json`
  - `section_plan.json`
  - `scores.json` (kept entries only)
  - `transcript.json`
  - Cut tolerance value with its explanatory rubric
  - Target runtime
- Build the prompt context before calling GPT:
  - The intent: `intent.md`.
  - The outline array (each entry with `section_id`, `title`, `goal`, `target_duration`).
  - A one-line status summary describing the current plan: total duration (sum of segment durations), visible cuts (count of non-sequential jumps), visible cuts per minute, target runtime, and the cut tolerance - along with brief explanations.
  - The current segment list in playback order, each as `{ "segment_id": int, "section_id": int, "text": str, "duration": float, "speed": float, "composite_score": float }`.
- Prompt objective: tell the model to treat the plan like a script ready for broadcast. It should read through as an experienced editor, imagining what the viewer hears and feels, and make high-impact adjustments that improve clarity, emotional flow, and informational value. Sections are guidance rather than handcuffs; it may drop or reorder segments, or bend section boundaries, when that produces a better overall experience.
- Stress that the stats are for awareness only - storytelling, intent alignment, and pacing are the primary goals. Use the numbers as sanity checks (stay near the target runtime, avoid egregious cut counts), not optimisation targets.
- Output `optimization.json`:

```json
{
  "segment_ids": [5, 9, 6, 12],
  "summary": "Trimmed redundant aside (segment 14) and moved the case vignette (9) ahead of the definition recap for narrative momentum. Runtime 8m12s; visible cuts steady at 1.8/min."
}
```

Raise an error if the model returns an empty list; the optimizer must produce at least one segment and explain its edits.

#### Stage G – Rendering

- Required inputs: `optimization.json` (segment ordering) and `transcript.json` (for start/end and file mapping).
- For each `segment_id`:
  - Resolve the source file path via `file_id`.
  - Pull precise `start`/`end`.
- Build an FFmpeg concat workflow:
  - Extract segments sequentially (use stream copy; provide a re-encode fallback if ffmpeg fails or output is tiny).
  - Log every command to `render.log`.
  - Concat to final `<stem>_cut.mp4` under `<output-dir>/<stem>/`.
  - If extraction/concat fails even after fallback, raise a fatal error.
- When `--plan-only` is set, skip rendering after writing `optimization.json` and print the path to that file.
- If the optimizer produced zero segments, raise `SystemExit` with an explanatory message - never emit an empty slate.

---

### 4. Shared Infrastructure

- **Logging**: `plan.log` and `render.log` are append-only, `[YYYY-MM-DD HH:MM:SS] message`.
- **HTTP**: single-shot calls to OpenRouter, no retries. If response status isn’t 2xx, raise `SystemExit` with the first 4k of body text. Write request/response into `debug/<tag>/` when `--debug` is set.
- **JSON handling**: strict `json.loads`. Any malformed or unexpected schema must abort the run.
- **Cache policy**: before each stage, derive earliest stage needing recompute based on missing artefacts or CLI arg changes (use `metadata.json` to persist previous args).
- **Utilities**: reuse command execution helper (`subprocess.Popen`) with text mode, `ensure_ffmpeg_available`, `try_get_openrouter_api_key`, etc.
- **Error strategy**: fail fast - do not silently recover. If a user edits artefacts into an invalid state, detect and abort with clear messages pointing to the relevant file/section.

---

### 5. Prompt Guidance Summaries

**Intent Stage**
- Emphasise that the model sees the full transcript text plus a user-supplied intent snippet.
- Instruct it to output:
  1. 2–3 bullets covering purpose, audience, tone.
  2. Four headings (**Essential**, **Important**, **Optional**, **Leave out**) each with concise bullets of themes.
- Remind it to keep the response under ~250 words and avoid referencing segment IDs or timings.

**Scoring Stage**
- Provide explicit rubric definitions:
  - `fact`: +1 truthful/precise, 0 neutral/uncertain, negative for incorrect or misleading statements.
  - `info`: 1 if the segment adds substantive new information toward the intent, 0 if filler.
  - `form`: 1 when delivery feels clean - minimal filler, coherent sentences, correct grammar, reasonable pacing. For pacing, use the supplied `speed` (chars/sec) to spot rushed or sluggish speech. **Target** \~12–18 cps (**12–16 cps is fine for denser content or a naturally slower style**); **soft penalty** at 10–12 (dragging) or 18–20 (rushed); **strong penalty** at <10 or >20.
  - `importance`: 1 if the segment is critical for the outlined intent, proportionally lower as relevance drops.
- Require a short `note` that justifies the scores (e.g., “Defines delirium succinctly,” “Confuses delirium with dementia”).
- Reinforce that output must be JSON matching:
  ```json
  {"segment_scores": [{"segment_id": int, "fact": float, "info": float, "form": float, "importance": float, "note": "..."}]}
  ```
- Set temperature to 0 for determinism.

**Structure Planning Stage**
- Clarify the task: design a section-by-section outline that delivers the intent within the target runtime, balancing flow (intro → core → wrap-up) and viewer engagement.
- Mention that segment samples include `duration`, `speed`, `composite_score`- use them to estimate pacing and material strength.
- Demand output as a JSON array of objects `{section_id, title, goal, target_duration}` with contiguous IDs starting at 1.
- Suggest including transitions (e.g., “transition from problem to solution”) in section goals.

**Section Mapping Stage**
- State the primary objective: assign the strongest available segments to each outline section so the story hits every goal, builds logically, and stays close to the target runtime.
- Use the per-segment duration, speed, and composite_score to judge pacing and quality; prefer natural adjacency when it supports the narrative, but reach for later segments when the section would otherwise miss key content.
- Treat visible cuts as a secondary constraint: remind the model that every non-sequential jump adds a cut and that the cut-tolerance value is a guardrail, not the only success metric.
- Require section_notes to capture trade-offs (e.g., summarise why a jump was necessary, or where pacing feels tight/loose).
- Output format: JSON array ordered like the outline, each entry {section_id, segment_ids, section_notes}.

**Optimization Stage**
- Tell the model it receives the outline (with goals and target durations), a trimmed intent summary, the target runtime, the cut-tolerance explanation, a quick status line (current total duration and visible cuts per minute), and the flattened segment table (`segment_id`, `section_id`, `text`, `duration`, `speed`, `composite_score`).
- Emphasise an editorial mindset: read the sequence like a final broadcast script, focus on narrative clarity, emotional cadence, and fulfilment of the stated intent; use the numbers only as sanity checks.
- Encourage precise, high-impact edits - trim redundancies, smooth transitions, highlight the strongest material - even if that means bending section boundaries. Ask the model to call out major departures from the outline.
- Require the output JSON to contain the final global segment order and a concise prose summary that explains the editorial choices and notes runtime/cut changes when relevant. Empty lists are invalid.
- Example summary guidance: "Removed segment 18 (duplicate definition). Moved Q&A before wrap-up for smoother flow. Runtime 8m12s; visible cuts steady at 1.9/min."

---

### 6. Rendering Notes

- Resolve FFmpeg/ffprobe paths once; raise descriptive errors if missing.
- Extraction commands should use `-ss <start>` and `-to <end>` with millisecond precision.
- Re-encode fallback uses `libx264`/`aac` (preset `veryfast`, CRF 18, 192k audio).
- Concat via demuxer text file; attempt stream copy first, re-encode full output if needed.
- Print the final video path to stdout on success.










