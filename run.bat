@echo off
REM Запуск бота на Windows.
REM Использование:
REM   run.bat --init   - тихий первый проход
REM   run.bat          - обычная работа
REM Скрипт сам создаёт окружение .venv-win и ставит зависимости, если их ещё нет.

REM Переходим в папку, где лежит сам скрипт (с учётом диска).
cd /d "%~dp0"

REM Проверяем, что конфигурация создана.
if not exist ".env" (
    echo [X] Файл .env не найден.
    echo     Создайте его из шаблона и заполните значения:
    echo         copy .env.example .env
    echo     ^(заполните CHANNELS, NOTIFY_BOT_TOKEN, NOTIFY_CHAT_ID, LLM_API_KEY, LLM_MODEL^)
    exit /b 1
)

if not exist ".venv-win\" (
    echo Окружение .venv-win не найдено - создаю и ставлю зависимости...
    py -m venv .venv-win
    .venv-win\Scripts\pip install --quiet --upgrade pip
    .venv-win\Scripts\pip install -r requirements.txt
)

REM Пробрасываем все аргументы (напр. --init) в main.py.
.venv-win\Scripts\python main.py %*
