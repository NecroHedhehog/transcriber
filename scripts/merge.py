"""
Склейка диаризованного транскрипта: объединяет короткие строки
одного спикера в блоки-абзацы.

Использование:
    python scripts/merge.py diarized.txt -o merged.txt
"""

import argparse


def merge_transcript(input_path: str, output_path: str) -> str:
    """Склеивает строки одного спикера в абзацы."""

    blocks = []
    current_speaker = None
    current_time = None
    current_texts = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[SPEAKER_") or line.startswith("[???]"):
                if current_speaker and current_texts:
                    blocks.append((current_speaker, current_time,
                                   " ".join(current_texts)))
                current_speaker = line.strip("[]")
                current_time = None
                current_texts = []

            elif line.startswith("[") and ":" in line[:6]:
                bracket_end = line.index("]")
                time_str = line[1:bracket_end]
                text = line[bracket_end + 1:].strip()
                if current_time is None:
                    current_time = time_str
                if text:
                    current_texts.append(text)

    if current_speaker and current_texts:
        blocks.append((current_speaker, current_time, " ".join(current_texts)))

    with open(output_path, "w", encoding="utf-8") as f:
        for speaker, time, text in blocks:
            f.write(f"{speaker} [{time}]: {text}\n\n")

    print(f"Готово! {len(blocks)} блоков → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Склейка строк по спикерам")
    parser.add_argument("input", help="Диаризованный транскрипт (из diarize.py)")
    parser.add_argument("-o", "--output", default="merged.txt",
                        help="Выходной файл (default: merged.txt)")
    args = parser.parse_args()

    merge_transcript(args.input, args.output)
