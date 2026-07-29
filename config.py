"""Чтение и валидация конфигурации из переменных окружения (.env).

Все секреты и настройки живут только в .env — в коде никаких хардкодов.
Здесь мы читаем переменные, приводим их к нужным типам, подставляем дефолты
и валидируем, что обязательные значения заданы.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


class ConfigError(Exception):
    """Понятная ошибка конфигурации (не задана обязательная переменная и т.п.)."""


# Поддерживаемые провайдеры LLM. Модель (LLM_MODEL) задаётся пользователем явно —
# дефолт не подставляется.
SUPPORTED_PROVIDERS = {"openai", "anthropic", "gemini"}


@dataclass(frozen=True)
class Config:
    """Неизменяемый снимок конфигурации приложения."""

    # --- Каналы для мониторинга ---
    channels: list[str] = field(default_factory=list)

    # --- Бот-уведомитель (Bot API) ---
    notify_bot_token: str = ""
    notify_chat_id: str = ""

    # --- LLM ---
    llm_provider: str = "openai"
    llm_model: str = ""
    llm_api_key: str = ""
    # Кастомный base_url для OpenAI-совместимых API (напр. OpenRouter).
    # Пусто = стандартный эндпоинт провайдера.
    llm_base_url: str = ""
    # Просить у модели строгий JSON (response_format). Отключите, если модель
    # на OpenRouter не поддерживает json_object (иначе запрос падает).
    llm_json_mode: bool = True
    # Требовать бесплатную модель: если true и модель на OpenRouter платная —
    # запуск завершается с ошибкой (проверка через OpenRouter /models).
    llm_require_free: bool = False

    # --- Логика подбора ---
    match_threshold: int = 70
    poll_interval_minutes: int = 30
    lookback_messages: int = 50
    # Окно первого цикла после запуска: за перерыв (ночь, выходные) могло выйти
    # больше постов, чем помещается в обычное окно, а не попавшее в него теряется
    # безвозвратно. Дальше окно возвращается к lookback_messages.
    startup_lookback_messages: int = 100
    first_run_silent: bool = False

    # --- Rate-limit (секунды между вызовами) ---
    llm_rate_delay: float = 1.0
    notify_rate_delay: float = 1.0

    # --- Реакция на сбои LLM ---
    # Сколько сбоев вызова подряд считать «провайдер лежит»: цикл прерывается,
    # необработанные посты остаются не-seen и берутся в следующем цикле.
    llm_failure_streak: int = 3
    # На сколько минут уснуть после такой серии сбоев (вместо обычного интервала).
    llm_backoff_minutes: int = 30

    # --- Веб-парсер t.me/s/ ---
    http_user_agent: str = ""
    http_timeout: float = 20.0

    # --- Пути ---
    resume_path: str = "resume.txt"
    db_path: str = "job_bot.sqlite3"
    log_path: str = "job_bot.log"


def _get_bool(name: str, default: bool) -> bool:
    """Читает булеву переменную окружения (true/1/yes/on)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _get_int(name: str, default: int) -> int:
    """Читает целочисленную переменную окружения с дефолтом."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    """Читает float-переменную окружения с дефолтом."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть числом, получено: {raw!r}") from exc


def _normalize_channel(raw: str) -> str:
    """Приводит канал к каноничному username (без @) или оставляет как есть.

    Поддерживает форматы: `@name`, `name`, `https://t.me/name`, `t.me/name`.
    Приватные инвайт-ссылки (`t.me/+hash`, `t.me/joinchat/...`) оставляем как есть —
    веб-ридер их не поддержит и пропустит с предупреждением.
    """
    value = raw.strip()
    if not value:
        return ""

    # Срезаем протокол и домен t.me, если это ссылка.
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    for prefix in ("t.me/", "telegram.me/"):
        if value.startswith(prefix):
            value = value[len(prefix):]

    # Инвайт-ссылки на приватные каналы — не трогаем.
    if value.startswith(("+", "joinchat/")):
        return value

    # Обычный публичный username — убираем ведущий @.
    return value.lstrip("@")


def _parse_channels(raw: str | None) -> list[str]:
    """Разбирает CHANNELS (список через запятую) в нормализованный список."""
    if not raw:
        return []
    result = []
    for part in raw.split(","):
        norm = _normalize_channel(part)
        if norm:
            result.append(norm)
    return result


def load_config(*, first_run_silent_override: bool | None = None) -> Config:
    """Загружает конфигурацию из окружения (.env) и валидирует обязательные поля.

    :param first_run_silent_override: если задан (напр. через CLI-флаг --init),
        имеет приоритет над переменной окружения FIRST_RUN_SILENT.
    :raises ConfigError: если не заданы обязательные переменные.
    """
    load_dotenv()

    missing: list[str] = []

    # --- Обязательные строковые переменные ---
    channels = _parse_channels(os.getenv("CHANNELS"))
    if not channels:
        missing.append("CHANNELS")

    notify_bot_token = os.getenv("NOTIFY_BOT_TOKEN", "").strip()
    if not notify_bot_token:
        missing.append("NOTIFY_BOT_TOKEN")

    notify_chat_id = os.getenv("NOTIFY_CHAT_ID", "").strip()
    if not notify_chat_id:
        missing.append("NOTIFY_CHAT_ID")

    llm_api_key = os.getenv("LLM_API_KEY", "").strip()
    if not llm_api_key:
        missing.append("LLM_API_KEY")

    # Модель указываем явно — дефолт не подставляем (особенно важно для OpenRouter,
    # где нужен id в формате провайдер/модель).
    llm_model = os.getenv("LLM_MODEL", "").strip()
    if not llm_model:
        missing.append("LLM_MODEL")

    # Если чего-то не хватает — сообщаем сразу обо всём, а не по одной.
    if missing:
        raise ConfigError(
            "Не заданы обязательные переменные окружения: "
            + ", ".join(missing)
            + ". Скопируйте .env.example в .env и заполните их."
        )

    # --- LLM провайдер и модель ---
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if llm_provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER={llm_provider!r} не поддерживается. "
            f"Доступны: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )
    llm_base_url = os.getenv("LLM_BASE_URL", "").strip()
    llm_json_mode = _get_bool("LLM_JSON_MODE", True)
    llm_require_free = _get_bool("LLM_REQUIRE_FREE", False)

    # --- FIRST_RUN_SILENT: приоритет у CLI-флага ---
    if first_run_silent_override is not None:
        first_run_silent = first_run_silent_override
    else:
        first_run_silent = _get_bool("FIRST_RUN_SILENT", False)

    return Config(
        channels=channels,
        notify_bot_token=notify_bot_token,
        notify_chat_id=notify_chat_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_json_mode=llm_json_mode,
        llm_require_free=llm_require_free,
        match_threshold=_get_int("MATCH_THRESHOLD", 70),
        poll_interval_minutes=_get_int("POLL_INTERVAL_MINUTES", 30),
        lookback_messages=_get_int("LOOKBACK_MESSAGES", 50),
        startup_lookback_messages=_get_int("STARTUP_LOOKBACK_MESSAGES", 100),
        first_run_silent=first_run_silent,
        llm_rate_delay=_get_float("LLM_RATE_DELAY", 1.0),
        notify_rate_delay=_get_float("NOTIFY_RATE_DELAY", 1.0),
        llm_failure_streak=_get_int("LLM_FAILURE_STREAK", 3),
        llm_backoff_minutes=_get_int("LLM_BACKOFF_MINUTES", 30),
        http_user_agent=os.getenv("HTTP_USER_AGENT", "").strip(),
        http_timeout=_get_float("HTTP_TIMEOUT", 20.0),
        resume_path=os.getenv("RESUME_PATH", "resume.txt").strip() or "resume.txt",
        db_path=os.getenv("DB_PATH", "job_bot.sqlite3").strip() or "job_bot.sqlite3",
        log_path=os.getenv("LOG_PATH", "job_bot.log").strip() or "job_bot.log",
    )
