# transcriber

Transcription pipeline for sociological focus groups and interviews.
One command: audio/video → speaker-labeled transcript with timestamps.

## What it does

```
Input:  recording.mp4 (54 min focus group, 8 speakers, phone recording)
Output: Модератор [00:00]: Добрый день, друзья. Сегодня мы поговорим о...
        Аня [00:41]: Я с философского факультета, направление рекламы...
        Юля [01:06]: С соцфака, четвёртый курс, однокурсники...
```

## Quick start

### Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- ffmpeg in PATH
- HuggingFace account (free) with accepted licenses:
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  - [pyannote/segmentation-community-1](https://huggingface.co/pyannote/segmentation-community-1)

### Install

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your HuggingFace token
```

### Run

Interactive mode (with menu):
```bash
python run.py
```

Automatic mode:
```bash
python run.py recording.mp4
python run.py recording.mp4 --skip-diarize
python run.py recording.mp4 --llm openai --type focus_group --mode clean
```

At first run, the script checks all dependencies and offers to install missing ones.

Drop your files into `input/`, results go to `output/`.

## Pipeline

```
Audio/Video
    │
    ▼
[ffmpeg] → 16kHz mono WAV
    │
    ├──► [Whisper large-v3 + Silero VAD] → timestamped transcript
    │
    ├──► [pyannote community-1] → speaker diarization map
    │
    ▼
[merge script] → speaker-labeled blocks
    │
    ▼
[LLM postprocessing] → names, punctuation, error correction (optional)
    │
    ▼
Final transcript (.txt)
```

**Silero VAD** filters silence and prevents Whisper hallucinations ("Продолжение следует" ×9, garbage in other languages). Double filtering: VAD marks silence AND Whisper's own `no_speech_prob > 0.7` must both trigger to drop a segment.

**Pyannote community-1** runs locally on GPU. Audio is read via Python `wave` module (workaround for torchcodec bug on Windows/CUDA 13). Speaker segments are aligned with transcript by timestamp overlap.

**LLM postprocessing** is optional (`--llm openai` or `--llm anthropic`). Prompts are stored as editable text files in `prompts/` — 4 templates for different research types and modes.

## Prompt templates

| File | Type | Mode |
|------|------|------|
| `interview_verbatim.txt` | Interview (1-3 speakers) | Keeps filler words (for discourse analysis) |
| `interview_clean.txt` | Interview (1-3 speakers) | Removes filler words (for content analysis) |
| `focus_group_verbatim.txt` | Focus group (4+ speakers) | Keeps filler words |
| `focus_group_clean.txt` | Focus group (4+ speakers) | Removes filler words |

Edit prompts with any text editor. Add new types by dropping a `.txt` file into `prompts/`.

## Benchmarks

Tested on real recordings:

**Focus group** (54 min, 8 speakers, phone mic):

| Method | Text quality | Speakers found | Cost/hour |
|--------|-------------|----------------|-----------|
| **This pipeline** | 8/10 → 9/10 after LLM | **8/8** | ~$0 |
| PyannoteAI STT Orchestration | 5-6/10 | 6/8 | $0.29 |
| PyannoteAI precision-2 | — | 6/8 | $0.17 |
| PyannoteAI community-1 cloud | — | 8/8 | $0.04 |
| ElevenLabs Scribe v2 | **10/10** | 7/8 | $0.40 |

**Expert interview** (66 min, 2 speakers, Zoom recording):

| Metric | Result |
|--------|--------|
| Speakers found | 2/2 + 1 ambiguous segment |
| Text quality | 9/10 (clean audio, minimal errors) |
| Processing time | ~18 min |

Key findings:
- Local pyannote community-1 separates speakers better than all tested cloud APIs for 6+ speakers
- Scribe v2 produces the best text quality but merges similar-sounding speakers
- Zoom recordings with good mic → near-perfect results without LLM postprocessing
- Phone recordings need LLM cleanup for production quality

## CLI options

```
python run.py <file> [options]

Options:
  -o, --output         Output file path (default: output/<name>_transcript.txt)
  -m, --model          Whisper model: large-v3, medium, small (default: large-v3)
  -l, --language       Audio language ISO code (default: ru)
  --llm                LLM provider: openai, anthropic (default: none)
  --type               Recording type: interview, focus_group (default: interview)
  --mode               Transcript mode: verbatim, clean (default: verbatim)
  --skip-diarize       Skip speaker diarization (text only)
```

## Tech notes

**GPU compatibility (RTX 5060 Ti / Blackwell / CUDA 13):**
- `openai-whisper` works via PyTorch directly — CUDA 13 compatible
- `faster-whisper` does NOT work — ctranslate2 is built for CUDA 12 only
- `torchcodec` broken on Windows with cu130 — workaround: read WAV via Python `wave` module
- Requires PyTorch 2.9.1+cu130

## Project structure

```
transcriber/
├── run.py              — main entry point (interactive menu + CLI)
├── scripts/
│   ├── transcribe.py   — Whisper + Silero VAD transcription
│   ├── diarize.py      — pyannote speaker diarization + alignment
│   └── merge.py        — merge short lines into speaker blocks
├── prompts/
│   ├── interview_verbatim.txt
│   ├── interview_clean.txt
│   ├── focus_group_verbatim.txt
│   └── focus_group_clean.txt
├── input/              — drop audio/video files here
├── output/             — results and intermediate files
├── .env.example        — template for API keys
├── .gitignore
├── requirements.txt
└── README.md
```

## Intermediate files

The pipeline saves all intermediate results in `output/`:

```
output/
├── recording_1_transcript.txt   ← Whisper + VAD (raw text with timestamps)
├── recording_2_diarized.txt     ← with speaker labels
├── recording_3_merged.txt       ← merged into blocks
└── recording_transcript.txt     ← final result
```

Useful for debugging, re-running only LLM step, or delivering text-only transcripts.

## License

MIT
