"""
Транскрибация аудио/видео с помощью Whisper large-v3 + Silero VAD.
VAD фильтрует тишину и убирает галлюцинации.

Использование:
    python scripts/transcribe.py input.mp4 -o transcript.txt
    python scripts/transcribe.py audio.wav -o transcript.txt --model large-v3
"""

import argparse
import os
import subprocess
import tempfile
import wave

import numpy as np
import torch
import whisper


def extract_audio(input_path: str, output_path: str) -> None:
    """Извлекает аудио из видео (16кГц, моно, PCM WAV)."""
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vn", "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", output_path, "-y"],
        check=True, capture_output=True,
    )


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Читает WAV через стандартный wave (обход torchcodec на Windows/CUDA 13)."""
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def get_vad_timestamps(audio: np.ndarray, sr: int = 16000) -> list[dict]:
    """Возвращает таймстемпы речевых сегментов через Silero VAD."""
    vad_model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]
    wav_tensor = torch.from_numpy(audio)
    return get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sr)


def is_speech(start_sec: float, end_sec: float, timestamps: list[dict],
              sr: int = 16000) -> bool:
    """Проверяет, попадает ли сегмент в речевой участок по VAD."""
    for ts in timestamps:
        ts_start = ts["start"] / sr
        ts_end = ts["end"] / sr
        if min(end_sec, ts_end) - max(start_sec, ts_start) > 0.3:
            return True
    return False


def transcribe(input_path: str, output_path: str, model_name: str = "large-v3",
               language: str = "ru") -> str:
    """Полный пайплайн: извлечение аудио → VAD → Whisper → фильтрация."""

    # Определяем, нужно ли извлекать аудио
    is_video = input_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm"))
    if is_video:
        audio_path = tempfile.mktemp(suffix=".wav")
        print("Извлекаю аудио из видео...")
        extract_audio(input_path, audio_path)
    else:
        audio_path = input_path

    try:
        print("Читаю аудио...")
        audio, sr = read_wav(audio_path)

        print("Анализирую голосовую активность (VAD)...")
        speech_timestamps = get_vad_timestamps(audio, sr)
        print(f"  Найдено {len(speech_timestamps)} речевых сегментов")

        print(f"Загружаю Whisper {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device=device)

        print("Транскрибирую...")
        result = model.transcribe(
            audio_path, language=language,
            beam_size=5, condition_on_previous_text=True,
        )

        # Фильтрация по VAD и no_speech_prob
        lines = []
        for seg in result["segments"]:
            if seg["no_speech_prob"] > 0.7:
                continue
            if not is_speech(seg["start"], seg["end"], speech_timestamps):
                continue
            text = seg["text"].strip()
            if not text or len(text) < 2:
                continue
            m = int(seg["start"] // 60)
            s = int(seg["start"] % 60)
            lines.append(f"[{m:02d}:{s:02d}] {text}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"\nГотово! {len(lines)} строк → {output_path}")
        return output_path

    finally:
        if is_video and os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Транскрибация с Whisper + VAD")
    parser.add_argument("input", help="Путь к аудио/видео файлу")
    parser.add_argument("-o", "--output", default="transcript.txt",
                        help="Путь для сохранения (default: transcript.txt)")
    parser.add_argument("-m", "--model", default="large-v3",
                        help="Модель Whisper (default: large-v3)")
    parser.add_argument("-l", "--language", default="ru",
                        help="Язык аудио (default: ru)")
    args = parser.parse_args()

    transcribe(args.input, args.output, args.model, args.language)
