"""
Полный пайплайн транскрибации.

Интерактивный режим (с меню):
    python run.py

Автоматический режим (с аргументами):
    python run.py video.mp4
    python run.py video.mp4 -o result.txt --llm openai
    python run.py video.mp4 --skip-diarize
"""

import argparse
import os
import sys
import time
import tempfile
import subprocess
import glob


# ── Автоустановка зависимостей ──────────────────────────────────────────────

DEPENDENCIES = {
    "whisper":        "openai-whisper",
    "pyannote.audio": "pyannote.audio",
    "dotenv":         "python-dotenv",
    "numpy":          "numpy",
    "torch":          "torch",
    "torchaudio":     "torchaudio",
}

OPTIONAL_DEPS = {
    "openai":    "openai",
    "anthropic": "anthropic",
}


def check_and_install():
    """Проверяет зависимости и устанавливает недостающие."""
    missing = []
    for module, package in DEPENDENCIES.items():
        try:
            __import__(module.split(".")[0])
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\nНе хватает пакетов: {', '.join(missing)}")
        answer = input("Установить? [Y/n] ").strip().lower()
        if answer in ("", "y", "д", "да", "yes"):
            for pkg in missing:
                print(f"  Устанавливаю {pkg}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                )
            print("Готово!\n")
        else:
            print("Установите вручную: pip install " + " ".join(missing))
            sys.exit(1)

    # Проверяем ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("\nffmpeg не найден! Установите:")
        print("  Windows: winget install ffmpeg")
        print("  Linux:   sudo apt install ffmpeg")
        print("  macOS:   brew install ffmpeg")
        sys.exit(1)

    # Проверяем .env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        example = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
        if os.path.exists(example):
            import shutil
            shutil.copy(example, env_path)
            print("Создан .env из шаблона — впишите туда HF_TOKEN\n")


def check_optional_dep(provider: str) -> bool:
    """Проверяет и ставит опциональную зависимость для LLM."""
    package = OPTIONAL_DEPS.get(provider)
    if not package:
        return False
    try:
        __import__(provider)
        return True
    except ImportError:
        answer = input(f"  Для LLM нужен пакет '{package}'. Установить? [Y/n] ").strip().lower()
        if answer in ("", "y", "д", "да", "yes"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"],
            )
            return True
        return False


# ── Интерактивное меню ──────────────────────────────────────────────────────

def find_media_files() -> list:
    """Ищет аудио/видео файлы в input/ и текущей директории."""
    extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm",
                  "*.wav", "*.mp3", "*.ogg", "*.m4a"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(ext))
    return sorted(set(files))


def interactive_menu() -> dict:
    """Интерактивное меню — возвращает настройки пайплайна."""

    print("\n" + "=" * 60)
    print("  🎙  Транскрибация аудио/видео")
    print("=" * 60)

    # 1. Выбор файла
    print("\n--- Файл ---\n")
    files = find_media_files()
    if files:
        print("Найдены файлы в текущей папке:")
        for i, f in enumerate(files, 1):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  {i}. {f}  ({size_mb:.0f} МБ)")
        print(f"  {len(files) + 1}. Ввести путь вручную")
        choice = input(f"\nВыберите [1-{len(files) + 1}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            input_path = files[int(choice) - 1]
        else:
            input_path = input("Путь к файлу: ").strip().strip('"')
    else:
        input_path = input("Путь к файлу: ").strip().strip('"')

    if not os.path.exists(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    # 2. Диаризация
    print("\n--- Диаризация (определение спикеров) ---\n")
    print("  1. С диаризацией — для фокус-групп и интервью")
    print("  2. Без диаризации — просто текст с таймкодами")
    choice = input("\nВыберите [1/2] (default: 1): ").strip()
    skip_diarize = choice == "2"

    # 3. Модель Whisper
    print("\n--- Модель Whisper ---\n")
    print("  1. large-v3  — лучшее качество, ~3 ГБ VRAM")
    print("  2. medium    — быстрее, ~2 ГБ VRAM")
    print("  3. small     — ещё быстрее, ~1 ГБ VRAM")
    choice = input("\nВыберите [1/2/3] (default: 1): ").strip()
    models = {"1": "large-v3", "2": "medium", "3": "small", "": "large-v3"}
    model = models.get(choice, "large-v3")

    # 4. Язык
    print("\n--- Язык ---\n")
    print("  1. Русский")
    print("  2. Английский")
    print("  3. Другой (ввести код)")
    choice = input("\nВыберите [1/2/3] (default: 1): ").strip()
    if choice == "2":
        language = "en"
    elif choice == "3":
        language = input("Код языка (например 'de', 'fr', 'es'): ").strip()
    else:
        language = "ru"

    # 5. LLM-постобработка
    print("\n--- LLM-постобработка (имена, пунктуация, ошибки) ---\n")
    print("  1. Пропустить — сделаю вручную потом")
    print("  2. OpenAI (GPT-4o-mini) — ~1 руб за транскрипт")
    print("  3. Anthropic (Claude) — ~5 руб за транскрипт")
    choice = input("\nВыберите [1/2/3] (default: 1): ").strip()
    llm = None
    if choice == "2":
        if check_optional_dep("openai"):
            llm = "openai"
    elif choice == "3":
        if check_optional_dep("anthropic"):
            llm = "anthropic"

    # 6. Выходной файл
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    default_output = os.path.join(output_dir, f"{base}_transcript.txt")
    print(f"\n--- Выходной файл ---\n")
    output = input(f"Имя файла (default: {default_output}): ").strip()
    if not output:
        output = default_output

    # Подтверждение
    print("\n" + "-" * 60)
    print(f"  Файл:        {input_path}")
    print(f"  Модель:      Whisper {model}")
    print(f"  Язык:        {language}")
    print(f"  Диаризация:  {'Нет' if skip_diarize else 'Да (pyannote)'}")
    print(f"  LLM:         {llm or 'Нет'}")
    print(f"  Результат:   {output}")
    print("-" * 60)

    confirm = input("\nЗапускаем? [Y/n] ").strip().lower()
    if confirm not in ("", "y", "д", "да", "yes"):
        print("Отменено.")
        sys.exit(0)

    return {
        "input_path": input_path,
        "output_path": output,
        "model": model,
        "language": language,
        "llm": llm,
        "skip_diarize": skip_diarize,
    }


# ── Основной пайплайн ──────────────────────────────────────────────────────

def ensure_wav(input_path: str) -> tuple:
    """Конвертирует в WAV если нужно."""
    if input_path.lower().endswith(".wav"):
        return input_path, False
    wav_path = tempfile.mktemp(suffix=".wav")
    print("Конвертирую в WAV...")
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vn", "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", wav_path, "-y"],
        check=True, capture_output=True,
    )
    return wav_path, True
    
    elapsed = time.time() - start_time
    print(f"  [{int(elapsed)}с] Аудио готово\n")

def llm_postprocess(text: str, provider: str) -> str:
    """Постобработка через LLM API."""
    from dotenv import load_dotenv
    load_dotenv()

    prompt = (
        "Ты — редактор транскриптов фокус-групп для социологических исследований.\n\n"
        "Выполни следующие задачи:\n\n"
        "1. ЗАМЕНА СПИКЕРОВ НА ИМЕНА\n"
        "В начале записи участники представляются. Определи кто есть кто и замени "
        "SPEAKER_XX на имена. Модератора обозначь как \"Модератор\". "
        "Если имя не удаётся определить — оставь \"Респондент N\".\n\n"
        "2. ЧИСТКА ТЕКСТА\n"
        "- Расставь пунктуацию\n"
        "- Исправь очевидные ошибки распознавания\n"
        "- НЕ удаляй слова-паразиты — они важны для анализа\n"
        "- НЕ добавляй слова которых не было\n\n"
        "3. ФОРМАТ\n"
        "Сохрани формат: Имя [ММ:СС]: текст реплики\n"
        "Каждая реплика — отдельный абзац.\n\n"
        "Вот транскрипт:\n\n"
    )

    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("  OPENAI_API_KEY не найден в .env, пропускаю")
            return text
        client = OpenAI(api_key=api_key)
        print("  Отправляю в GPT-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt + text}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("  ANTHROPIC_API_KEY не найден в .env, пропускаю")
            return text
        client = anthropic.Anthropic(api_key=api_key)
        print("  Отправляю в Claude...")
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt + text}],
        )
        return response.content[0].text

    return text


def run(input_path: str, output_path: str, model: str = "large-v3",
        language: str = "ru", llm: str = None, skip_diarize: bool = False):
    """Полный пайплайн."""

    # Импортируем скрипты
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    sys.path.insert(0, scripts_dir)
    from transcribe import transcribe
    from diarize import diarize
    from merge import merge_transcript

    start_time = time.time()
    total_steps = 1 if skip_diarize else 3
    if llm:
        total_steps += 1

    print(f"\n{'='*60}")
    print(f"  Обработка: {os.path.basename(input_path)}")
    print(f"{'='*60}\n")

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    transcript_path = os.path.join(out_dir, f"{base}_1_transcript.txt")
    diarized_path = os.path.join(out_dir, f"{base}_2_diarized.txt")
    merged_path = os.path.join(out_dir, f"{base}_3_merged.txt")

    wav_path, cleanup_wav = ensure_wav(input_path)

    try:
        step = 1

        # Шаг 1: Транскрибация
        print(f"\n--- Шаг {step}/{total_steps}: Транскрибация (Whisper + VAD) ---\n")
        transcribe(wav_path, transcript_path, model, language)
        step += 1

        if skip_diarize:
            final_text = open(transcript_path, "r", encoding="utf-8").read()
        else:
            # Шаг 2: Диаризация
            print(f"\n--- Шаг {step}/{total_steps}: Диаризация (pyannote) ---\n")
            diarize(wav_path, transcript_path, diarized_path)
            step += 1

            # Шаг 3: Склейка
            print(f"\n--- Шаг {step}/{total_steps}: Склейка ---\n")
            merge_transcript(diarized_path, merged_path)
            final_text = open(merged_path, "r", encoding="utf-8").read()
            step += 1

        # LLM-постобработка
        if llm:
            print(f"\n--- Шаг {step}/{total_steps}: LLM-постобработка ({llm}) ---\n")
            final_text = llm_postprocess(final_text, llm)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print(f"\n{'='*60}")
        print(f"  Готово! {minutes}м {seconds}с")
        print(f"  Результат: {os.path.abspath(output_path)}")
        print(f"{'='*60}\n")

    finally:
        if cleanup_wav and os.path.exists(wav_path):
            os.remove(wav_path)


# ── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_and_install()

    # Если передан файл — автоматический режим
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        parser = argparse.ArgumentParser(description="Транскрибация — полный пайплайн")
        parser.add_argument("input", help="Аудио или видео файл")
        parser.add_argument("-o", "--output", default="result.txt")
        parser.add_argument("-m", "--model", default="large-v3")
        parser.add_argument("-l", "--language", default="ru")
        parser.add_argument("--llm", choices=["openai", "anthropic"], default=None)
        parser.add_argument("--skip-diarize", action="store_true")
        args = parser.parse_args()
        run(args.input, args.output, args.model, args.language,
            args.llm, args.skip_diarize)
    else:
        # Без аргументов — интерактивное меню
        config = interactive_menu()
        run(**config)
