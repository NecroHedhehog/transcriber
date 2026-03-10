"""
Диаризация: определяет кто когда говорит и совмещает с транскриптом.
Использует pyannote community-1 (бесплатная, требует HuggingFace токен).

Использование:
    python scripts/diarize.py input.wav transcript.txt -o diarized.txt

Требуется:
    - HuggingFace токен в .env (HF_TOKEN=hf_...)
    - Принятые лицензии на huggingface.co:
      * pyannote/speaker-diarization-community-1
      * pyannote/segmentation-community-1
"""

import argparse
import os
import subprocess
import tempfile
import wave

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


def load_hf_token() -> str:
    """Загружает HuggingFace токен из .env или переменной окружения."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN не найден. Создайте .env файл с HF_TOKEN=hf_ваш_токен\n"
            "Получить токен: https://huggingface.co/settings/tokens"
        )
    return token


def ensure_audio(input_path: str) -> tuple[str, bool]:
    """Если передано видео — извлекает аудио. Возвращает (путь, нужно_удалить)."""
    if input_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
        audio_path = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-vn", "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", audio_path, "-y"],
            check=True, capture_output=True,
        )
        return audio_path, True
    return input_path, False


def read_wav_as_tensor(path: str) -> tuple[torch.Tensor, int]:
    """Читает WAV в torch tensor (обход torchcodec на Windows/CUDA 13)."""
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return waveform, sr


def run_diarization(waveform: torch.Tensor, sr: int, hf_token: str):
    """Запускает pyannote community-1 и возвращает результат диаризации."""
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    result = pipeline({"waveform": waveform, "sample_rate": sr})
    return result.speaker_diarization


def parse_transcript(path: str) -> list[dict]:
    """Парсит транскрипт формата [MM:SS] текст."""
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            bracket_end = line.index("]")
            time_str = line[1:bracket_end]
            text = line[bracket_end + 1:].strip()
            parts = time_str.split(":")
            start_sec = int(parts[0]) * 60 + int(parts[1])
            segments.append({"start": start_sec, "text": text})
    return segments


def get_speaker(time_sec: float, diarization, window: float = 2.0) -> str:
    """Находит спикера для данного момента времени."""
    best_speaker = "???"
    best_overlap = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = min(time_sec + window, turn.end) - max(time_sec, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def diarize(audio_path: str, transcript_path: str, output_path: str) -> str:
    """Полный пайплайн: аудио + транскрипт → транскрипт с разметкой спикеров."""

    hf_token = load_hf_token()
    audio_file, cleanup = ensure_audio(audio_path)

    try:
        print("Читаю аудио...")
        waveform, sr = read_wav_as_tensor(audio_file)

        print("Загружаю модель диаризации...")
        diarization = run_diarization(waveform, sr, hf_token)

        print("Совмещаю транскрипт со спикерами...")
        segments = parse_transcript(transcript_path)

        speakers_found = set()
        with open(output_path, "w", encoding="utf-8") as f:
            prev_speaker = None
            for seg in segments:
                speaker = get_speaker(seg["start"], diarization)
                speakers_found.add(speaker)
                m = seg["start"] // 60
                s = seg["start"] % 60
                if speaker != prev_speaker:
                    f.write(f"\n[{speaker}]\n")
                    prev_speaker = speaker
                f.write(f"[{m:02d}:{s:02d}] {seg['text']}\n")

        print(f"\nГотово! → {output_path}")
        print(f"Найдено спикеров: {len(speakers_found)}")
        for sp in sorted(speakers_found):
            print(f"  {sp}")

        return output_path

    finally:
        if cleanup and os.path.exists(audio_file):
            os.remove(audio_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Диаризация + совмещение с транскриптом")
    parser.add_argument("audio", help="Путь к аудио/видео файлу")
    parser.add_argument("transcript", help="Путь к транскрипту (из transcribe.py)")
    parser.add_argument("-o", "--output", default="diarized.txt",
                        help="Выходной файл (default: diarized.txt)")
    args = parser.parse_args()

    diarize(args.audio, args.transcript, args.output)
