# CLAUDE.md — контекст проекта для Claude Code

Телеграм-бот подбора вакансий: читает посты из публичных Telegram-каналов,
отсеивает не-вакансии, оценивает остальные через LLM на соответствие резюме
(0–100) и присылает подходящие в личку через бота-уведомителя. Работает по
расписанию, один пост дважды не обрабатывает.

## Архитектура
- `main.py` — оркестрация + планировщик (asyncio-цикл, graceful shutdown).
- `config.py` — чтение/валидация `.env` в dataclass `Config`.
- `telegram_reader.py` — `WebReader`: чтение каналов через веб-превью `t.me/s/`.
- `prefilter.py` — эвристика «вакансия / не вакансия» (экономия токенов).
- `llm_client.py` — абстракция LLM (OpenAI/Anthropic/Gemini) + `PROMPT_TEMPLATE`.
- `notifier.py` — отправка уведомлений через Telegram Bot API (httpx).
- `db.py` — SQLite: `seen_posts`, `matches`.
- `resume.txt` / `positions.txt` — критерии оценки (см. `RESUME_PATH`); реальные
  файлы gitignored, в репозитории только шаблоны `*.example`.

## Ключевые решения (важно не сломать)
- **Чтение без api-ключа Telegram.** Каналы читаются веб-парсером
  `https://t.me/s/<username>` (httpx + BeautifulSoup), НЕ через Telethon/MTProto.
  Следствие: работают только **публичные** каналы с веб-превью; приватные и
  инвайт-ссылки (`t.me/+…`, `joinchat/…`) не поддерживаются и пропускаются.
- **LLM через OpenRouter (OpenAI-совместимый режим).** `LLM_PROVIDER=openai` +
  `LLM_BASE_URL=https://openrouter.ai/api/v1`, ключ `sk-or-…`. Модель — в формате
  OpenRouter (`провайдер/модель`, у бесплатных суффикс `:free`).
- **`LLM_MODEL` обязателен** — дефолт не подставляется (иначе `ConfigError`).
- **Для бесплатных моделей `LLM_JSON_MODE=false`** — многие не поддерживают
  `response_format` и падают. Парсер JSON в `llm_client._parse_json` устойчив и
  без строгого режима.
- **`LLM_REQUIRE_FREE=true`** — при старте проверяет через OpenRouter `/models`,
  что модель бесплатна; платная → `ConfigError` и выход (`verify_free_model`).
- **Жёсткое правило в промпте:** если формат вакансии — гибрид/очно и полная
  удалёнка невозможна, LLM возвращает `score=0` (см. `PROMPT_TEMPLATE`).
- **Формат уведомления:** только оценка + текст вакансии без изменений + ссылка
  (`notifier._build_message`); обрезка только на лимите Telegram 4096.

## Запуск
Из папки проекта, интерпретатор из venv:
```bash
.venv/bin/python main.py --init   # тихий первый проход (пометить старые посты seen)
.venv/bin/python main.py          # обычная постоянная работа
```
`main.py --init` НЕ завершается: после тихого прохода переходит в обычный режим и
работает дальше в том же процессе. Остановка — Ctrl+C.

## Готчи
- **venv:** зависимости ставить в `.venv` (`.venv/bin/pip install -r requirements.txt`),
  не в системный Python. Запуск — `.venv/bin/python …`.
- **Код и `.env` читаются один раз при старте** — после правок нужен перезапуск.
- **Лимиты free-моделей OpenRouter:** суточный кап запросов; при `429` увеличить
  `LLM_RATE_DELAY` / уменьшить `LOOKBACK_MESSAGES` / пополнить баланс.
- **Каналы `@it_jobs` и подобные** могут не читаться (нет веб-превью) — проверять
  через `https://t.me/s/<username>` в браузере.

## Обязательные переменные `.env`
`CHANNELS`, `NOTIFY_BOT_TOKEN`, `NOTIFY_CHAT_ID`, `LLM_API_KEY`, `LLM_MODEL`
(+ `LLM_BASE_URL` для OpenRouter). Остальное — с дефолтами (см. `.env.example`).

## Git / приватность (проект готовится к публикации на GitHub)
- **Не коммитить** (в `.gitignore`): `.env`, `resume.txt`, `positions.txt`,
  `.venv/`, `job_bot.sqlite3*`, `job_bot.log`, `.claude/settings.local.json`.
- **Коммитить** только шаблоны: `.env.example`, `resume.txt.example`,
  `positions.txt.example` — в них ТОЛЬКО плейсхолдеры, без реальных значений.
- В `README.md` и `*.example` не должно быть реальных данных: chat_id, токена бота,
  ключа OpenRouter, реального списка каналов. Использовать placeholder'ы.
- `CLAUDE.md` — коммитить (общий контекст, без секретов). Обновлять вручную после
  значимых изменений поведения бота.
