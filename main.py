"""Точка входа: оркестрация цикла подбора вакансий + планировщик.

Один цикл:
  1. читаем новые посты из каналов;
  2. каждый пост сразу помечаем seen (чтобы не обрабатывать дважды даже при сбое);
  3. дешёвый префильтр «вакансия / не вакансия»;
  4. вакансии отдаём в LLM на оценку;
  5. score >= порога → сохраняем и шлём уведомление;
  6. логируем статистику.

Планировщик — простой asyncio-цикл: run_cycle() каждые POLL_INTERVAL_MINUTES.
Graceful shutdown по Ctrl+C / SIGTERM.

Первый запуск: с флагом --init (или FIRST_RUN_SILENT=true) все старые посты
помечаются seen без оценки и отправки — чтобы не спамить старыми вакансиями.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import httpx

import db
from config import Config, ConfigError, load_config
from llm_client import make_llm_client, verify_free_model
from notifier import send_match
from prefilter import looks_like_vacancy
from telegram_reader import WebReader

logger = logging.getLogger(__name__)


def setup_logging(log_path: str) -> None:
    """Настраивает логирование в консоль и файл."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_resume(path: str) -> str:
    """Читает резюме из файла (один раз при старте). Кэшируется вызывающим кодом."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        raise ConfigError(
            f"Файл резюме не найден: {path}. Создайте resume.txt и вставьте своё резюме."
        )
    if not text:
        raise ConfigError(f"Файл резюме {path} пуст. Вставьте текст резюме.")
    return text


async def run_cycle(
    config: Config,
    reader: WebReader,
    llm,
    http_client: httpx.AsyncClient,
    resume_text: str,
) -> None:
    """Выполняет один цикл обработки. Ведёт статистику и логирует её в конце."""
    checked = 0        # всего проверено новых постов
    passed_filter = 0  # прошло префильтр
    scored = 0         # оценено через LLM
    sent = 0           # отправлено уведомлений

    posts = await reader.fetch_new_posts(
        channels=config.channels,
        lookback=config.lookback_messages,
        is_seen=db.is_seen,
    )

    for post in posts:
        checked += 1
        channel, msg_id = post["channel"], post["msg_id"]

        # Помечаем seen СРАЗУ — чтобы падение ниже не привело к повторной обработке.
        db.mark_seen(channel, msg_id)

        # Тихий первый запуск: только помечаем seen, ничего не оцениваем/шлём.
        if config.first_run_silent:
            continue

        # Дешёвый фильтр перед LLM, чтобы не жечь токены.
        if not looks_like_vacancy(post["text"]):
            continue
        passed_filter += 1

        # Оценка через LLM (синхронный клиент — уводим в поток, чтобы не блокировать loop).
        result = await asyncio.to_thread(llm.score_vacancy, resume_text, post["text"])
        scored += 1
        await asyncio.sleep(config.llm_rate_delay)  # rate-limit между вызовами LLM

        if result["score"] < config.match_threshold:
            continue

        # Подходящая вакансия: сохраняем и уведомляем.
        db.save_match(
            channel=channel,
            msg_id=msg_id,
            score=result["score"],
            reason=result["reason"],
            role=result["role"],
            vacancy_text=post["text"],
        )
        match = {
            "score": result["score"],
            "role": result["role"],
            "reason": result["reason"],
            "channel": channel,
            "vacancy_text": post["text"],
            "url": post["url"],
        }
        await send_match(config, match, http_client)
        sent += 1
        await asyncio.sleep(config.notify_rate_delay)  # rate-limit между отправками

    mode = " (тихий первый запуск)" if config.first_run_silent else ""
    logger.info(
        "Цикл завершён%s: проверено=%d, вакансий(префильтр)=%d, оценено=%d, отправлено=%d",
        mode, checked, passed_filter, scored, sent,
    )


async def main_loop(config: Config) -> None:
    """Инициализация ресурсов и бесконечный цикл-планировщик с graceful shutdown."""
    db.init_db(config.db_path)
    resume_text = load_resume(config.resume_path)
    llm = make_llm_client(config)

    reader = WebReader(user_agent=config.http_user_agent, timeout=config.http_timeout)
    await reader.start()

    http_client = httpx.AsyncClient()

    # Флаг остановки + обработка сигналов для чистого выхода.
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        logger.info("Получен сигнал остановки, завершаемся...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # add_signal_handler недоступен на некоторых платформах (напр. Windows).
            pass

    interval = config.poll_interval_minutes * 60
    try:
        while not stop_event.is_set():
            try:
                await run_cycle(config, reader, llm, http_client, resume_text)
            except Exception as exc:  # noqa: BLE001 — цикл не должен падать целиком
                logger.exception("Ошибка в цикле обработки: %s", exc)

            # После первого тихого прохода дальше работаем в обычном режиме.
            if config.first_run_silent:
                logger.info("Первый тихий проход выполнен, переходим в обычный режим.")
                config = _clear_first_run_silent(config)
                interval = config.poll_interval_minutes * 60

            # Ждём следующий цикл, но просыпаемся сразу по сигналу остановки.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass  # штатное истечение интервала — новый цикл
    finally:
        logger.info("Закрываем ресурсы...")
        await reader.disconnect()
        await http_client.aclose()
        db.close_db()
        logger.info("Остановлено.")


def _clear_first_run_silent(config: Config) -> Config:
    """Возвращает копию конфигурации с выключенным тихим режимом."""
    from dataclasses import replace

    return replace(config, first_run_silent=False)


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Telegram-бот подбора вакансий по резюме.")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Тихий первый запуск: пометить старые посты seen без оценки и отправки.",
    )
    return parser.parse_args()


def main() -> None:
    """Синхронная точка входа."""
    args = parse_args()

    # Загружаем конфиг: флаг --init имеет приоритет над FIRST_RUN_SILENT.
    override = True if args.init else None
    try:
        config = load_config(first_run_silent_override=override)
    except ConfigError as exc:
        # Логирование ещё не настроено — печатаем в stderr.
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config.log_path)
    logger.info("Запуск бота. Каналов: %d, порог: %d, интервал: %d мин.",
                len(config.channels), config.match_threshold, config.poll_interval_minutes)

    # Опциональная проверка: модель должна быть бесплатной (LLM_REQUIRE_FREE=true).
    try:
        verify_free_model(config)
    except ConfigError as exc:
        logger.error("Проверка модели не пройдена: %s", exc)
        sys.exit(1)

    try:
        asyncio.run(main_loop(config))
    except KeyboardInterrupt:
        logger.info("Прервано пользователем (Ctrl+C).")


if __name__ == "__main__":
    main()
