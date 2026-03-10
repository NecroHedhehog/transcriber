# transcriber

Transcription pipeline for sociological focus groups and interviews. Produces speaker-labeled transcripts from audio/video recordings.

## What it does

Raw video/audio → speaker-labeled transcript with timestamps, ready for qualitative research.

```
Input:  VID_20260307.mp4 (54 min focus group, 8 speakers, phone recording)
Output: Модератор [00:00]: Добрый день, друзья. Сегодня мы поговорим о...
        Аня [00:41]: Я с философского факультета, направление рекламы...
        Юля [01:06]: С соцфака, четвёртый курс, однокурсники...
```

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
[LLM postprocessing] → names, punctuation, error correction
    │
    ▼
Final transcript (.txt / .docx)
```

## Quick start

### Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- ffmpeg installed and in PATH
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

```bash
# Step 1: Transcribe (Whisper + VAD)
python scripts/transcribe.py recording.mp4 -o transcript.txt

# Step 2: Diarize (speaker labels)
python scripts/diarize.py recording.mp4 transcript.txt -o diarized.txt

# Step 3: Merge short lines into blocks
python scripts/merge.py diarized.txt -o merged.txt

# Step 4: LLM postprocessing (manual — paste into Claude/GPT with prompt from prompts/postprocess.md)
```

## Benchmarks

Tested on a real focus group recording (54 min, 8 speakers, phone mic):

| Method | Text quality | Speakers found | Cost/hour |
|--------|-------------|----------------|-----------|
| **Local pipeline (this repo)** | 8/10 → 9/10 after LLM | **8/8** | ~$0 |
| PyannoteAI STT Orchestration | 5-6/10 | 6/8 | $0.29 |
| PyannoteAI precision-2 (diarization only) | — | 6/8 | $0.17 |
| PyannoteAI community-1 cloud | — | 8/8 | $0.04 |
| ElevenLabs Scribe v2 | **10/10** | 7/8 | $0.40 |

Key finding: local pyannote community-1 consistently separates speakers better than cloud APIs for 6+ speaker recordings. Scribe v2 produces the best text but merges similar-sounding speakers.

## Tech notes

**GPU compatibility (RTX 5060 Ti / Blackwell / CUDA 13):**
- `openai-whisper` works via PyTorch directly — CUDA 13 compatible
- `faster-whisper` does NOT work — ctranslate2 is built for CUDA 12 only
- `torchcodec` broken on Windows with cu130 — workaround: read WAV via Python `wave` module
- Requires PyTorch 2.9.1+cu130

## Project structure

```
focusgroup-transcriber/
├── scripts/
│   ├── transcribe.py   — Whisper + Silero VAD transcription
│   ├── diarize.py      — pyannote speaker diarization + alignment
│   └── merge.py        — merge short lines into speaker blocks
├── prompts/
│   └── postprocess.md  — LLM prompts for transcript cleanup
├── .env.example        — template for API keys
├── .gitignore
├── requirements.txt
└── README.md
```

## License

MIT
