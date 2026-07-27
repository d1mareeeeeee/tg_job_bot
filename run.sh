#!/usr/bin/env bash
# Запуск бота на macOS / Linux.
# Использование:
#   ./run.sh --init   # тихий первый проход
#   ./run.sh          # обычная работа
# Скрипт сам создаёт окружение .venv и ставит зависимости, если их ещё нет.
set -e

# Переходим в папку, где лежит сам скрипт (можно запускать откуда угодно).
cd "$(dirname "$0")"

# Проверяем, что конфигурация создана.
if [ ! -f ".env" ]; then
    echo "❌ Файл .env не найден."
    echo "   Создайте его из шаблона и заполните значения:"
    echo "       cp .env.example .env"
    echo "   (заполните CHANNELS, NOTIFY_BOT_TOKEN, NOTIFY_CHAT_ID, LLM_API_KEY, LLM_MODEL)"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Окружение .venv не найдено — создаю и ставлю зависимости..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Пробрасываем все аргументы (напр. --init) в main.py.
exec .venv/bin/python main.py "$@"
