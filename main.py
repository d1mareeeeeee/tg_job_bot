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
from dataclasses import replace
from typing import NamedTuple

import httpx

import db
from config import Config, ConfigError, load_config
from llm_client import LLMCallError, LLMParseError, make_llm_client, verify_free_model
from notifier import send_error, send_match
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


def _short(text: str, limit: int) -> str:
    """Однострочная выжимка текста для лога: без переносов, не длиннее limit."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


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


class CycleResult(NamedTuple):
    """Итог цикла для планировщика."""

    # Провайдер похоже лежит (серия сбоев подряд) — стоит уйти в длинную паузу
    # вместо обычного интервала.
    backoff: bool = False
    # Ни один канал не прочитан (нет доступа к t.me: сеть, файрвол, выключенный VPN).
    # Такой цикл не считается удачным: стартовое окно сужать нельзя.
    channels_unreadable: bool = False


async def run_cycle(
    config: Config,
    reader: WebReader,
    llm,
    http_client: httpx.AsyncClient,
    resume_text: str,
) -> CycleResult:
    """Выполняет один цикл обработки. Ведёт статистику и логирует её в конце.

    Пост помечается seen только когда он действительно обработан. Если LLM не
    ответила (`LLMCallError`), пост остаётся не-seen и вернётся в следующем цикле.

    :return: `CycleResult` — нужна ли длинная пауза и удалось ли вообще прочитать каналы.
    """
    checked = 0        # всего проверено новых постов
    passed_filter = 0  # прошло префильтр
    scored = 0         # успешно оценено через LLM
    call_errors = 0    # сбоев вызова LLM (посты вернутся в следующем цикле)
    parse_errors = 0   # ответов LLM, которые не удалось разобрать
    sent = 0           # отправлено уведомлений

    fail_streak = 0    # сбоев вызова подряд
    backoff = False    # провайдер лежит — просим вызывающего уйти в паузу
    error_notices = 0  # отправлено сообщений об ошибках (ограничиваем флуд)

    posts, channels_unreadable = await reader.fetch_new_posts(
        channels=config.channels,
        lookback=config.lookback_messages,
        is_seen=db.is_seen,
    )

    for post in posts:
        checked += 1
        channel, msg_id = post["channel"], post["msg_id"]

        # Тихий первый запуск: только помечаем seen, ничего не оцениваем/шлём.
        if config.first_run_silent:
            db.mark_seen(channel, msg_id)
            continue

        # Дешёвый фильтр перед LLM, чтобы не жечь токены.
        if not looks_like_vacancy(post["text"]):
            db.mark_seen(channel, msg_id)
            logger.info("Префильтр отсеял %s: %s", post["url"], _short(post["text"], 80))
            continue
        passed_filter += 1

        # Оценка через LLM (синхронный клиент — уводим в поток, чтобы не блокировать loop).
        parse_failed = False
        result = None
        try:
            result = await asyncio.to_thread(llm.score_vacancy, resume_text, post["text"])
        except LLMCallError as exc:
            # Провайдер не ответил — seen НЕ ставим, вернёмся к посту позже.
            call_errors += 1
            fail_streak += 1
            logger.warning(
                "Пост %s/%s не оценён (%s) — повторим в следующем цикле.",
                channel, msg_id, exc,
            )
            going_to_pause = fail_streak >= config.llm_failure_streak
            if going_to_pause:
                logger.warning(
                    "Подряд %d сбоев вызова LLM — прерываем цикл, необработанные посты "
                    "останутся на следующий раз.", fail_streak,
                )
                backoff = True

            # Сообщаем пользователю, но не более llm_failure_streak раз за цикл:
            # при чередовании сбоев и успехов серия не растёт, а сообщений было бы много.
            if error_notices < config.llm_failure_streak:
                error_notices += 1
                last_notice = error_notices >= config.llm_failure_streak
                await send_error(
                    config,
                    http_client,
                    title="⚠️ Сбой вызова LLM",
                    details=[
                        f"Попытка {fail_streak} из {config.llm_failure_streak}",
                        f"Пост: {post['url']}",
                        f"Ошибка: {exc}",
                        f"Пауза {config.llm_backoff_minutes} мин до следующей попытки."
                        if going_to_pause else "",
                        "Дальнейшие сбои в этом цикле — только в лог."
                        if last_notice and not going_to_pause else "",
                    ],
                )
                await asyncio.sleep(config.notify_rate_delay)

            await asyncio.sleep(config.llm_rate_delay)
            if going_to_pause:
                break
            continue
        except LLMParseError as exc:
            # Провайдер жив (значит серия сбоев прервана), но ответ не разобран:
            # оценки нет — доносим вакансию до пользователя с пометкой.
            parse_errors += 1
            fail_streak = 0
            parse_failed = True
            logger.warning("Ответ LLM по посту %s/%s не разобран: %s", channel, msg_id, exc)
        else:
            scored += 1
            fail_streak = 0
            # Оценку пишем в лог всегда. В matches попадают только прошедшие порог,
            # поэтому без этой строки нельзя ответить на вопрос «почему ничего не
            # пришло»: непонятно, вакансии были неподходящие или что-то сломалось.
            logger.info(
                "Оценка %d (порог %d): %s — %s | %s",
                result["score"], config.match_threshold, post["url"],
                result["role"] or "роль не указана", _short(result["reason"], 160),
            )

        await asyncio.sleep(config.llm_rate_delay)  # rate-limit между вызовами LLM

        # Пост обработан (оценён или признан неразбираемым) — закрываем его.
        # Ставим seen ДО отправки: сбой отправки не должен приводить к дублю.
        db.mark_seen(channel, msg_id)

        if not parse_failed and result["score"] < config.match_threshold:
            continue

        # Подходящая вакансия (или парс-сбой, который шлём мимо порога):
        # сохраняем и уведомляем.
        db.save_match(
            channel=channel,
            msg_id=msg_id,
            score=0 if parse_failed else result["score"],
            reason="parse_error" if parse_failed else result["reason"],
            role="" if parse_failed else result["role"],
            vacancy_text=post["text"],
        )
        match = {
            "score": 0 if parse_failed else result["score"],
            "role": "" if parse_failed else result["role"],
            "reason": "parse_error" if parse_failed else result["reason"],
            "channel": channel,
            "vacancy_text": post["text"],
            "url": post["url"],
            "parse_error": parse_failed,
        }
        await send_match(config, match, http_client)
        sent += 1
        await asyncio.sleep(config.notify_rate_delay)  # rate-limit между отправками

    mode = " (тихий первый запуск)" if config.first_run_silent else ""
    logger.info(
        "Цикл завершён%s: проверено=%d, вакансий(префильтр)=%d, оценено=%d, "
        "сбоев LLM=%d, ошибок парсинга=%d, отправлено=%d",
        mode, checked, passed_filter, scored, call_errors, parse_errors, sent,
    )
    return CycleResult(backoff=backoff, channels_unreadable=channels_unreadable)


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
    # Сигнатура последнего падения — чтобы не слать одно и то же каждый цикл.
    last_crash: str | None = None
    # Уже сообщили, что каналы не читаются — не повторяем это каждый цикл.
    unreadable_notified = False
    # Первый цикл после запуска смотрит канал шире: за перерыв (ночь, выходные)
    # могло выйти больше постов, чем помещается в обычное окно, а всё, что в него
    # не попало, теряется безвозвратно — окно движется только вперёд.
    first_cycle = True
    try:
        while not stop_event.is_set():
            backoff = False
            channels_unreadable = False  # цикл упал — считаем, что не знаем
            if first_cycle and config.startup_lookback_messages > config.lookback_messages:
                cycle_config = replace(
                    config, lookback_messages=config.startup_lookback_messages
                )
                logger.info(
                    "Окно расширено до %d постов на канал (сузим до %d после цикла, "
                    "который прочитает каналы).",
                    config.startup_lookback_messages, config.lookback_messages,
                )
            else:
                cycle_config = config
            try:
                result = await run_cycle(cycle_config, reader, llm, http_client, resume_text)
                backoff = result.backoff
                channels_unreadable = result.channels_unreadable
            except Exception as exc:  # noqa: BLE001 — цикл не должен падать целиком
                logger.exception("Ошибка в цикле обработки: %s", exc)
                crash = f"{type(exc).__name__}: {exc}"
                # Повторяющееся падение шлём один раз: иначе постоянная ошибка
                # писала бы в личку каждые POLL_INTERVAL_MINUTES бесконечно.
                if crash != last_crash:
                    await send_error(
                        config,
                        http_client,
                        title="🛑 Непредвиденная ошибка цикла",
                        details=[
                            crash,
                            f"Бот продолжит работу, следующая попытка через "
                            f"{config.poll_interval_minutes} мин.",
                        ],
                    )
                else:
                    logger.info("Та же ошибка, что и в прошлом цикле — уведомление не дублируем.")
                last_crash = crash
            else:
                last_crash = None  # цикл прошёл — следующее падение снова уведомим
                # Окно сужаем только после удачного цикла: если он упал или не прочитал
                # ни одного канала, посты не помечены seen, и на повторе широкое окно
                # ещё пригодится. Иначе единственная защита от потери постов за простой
                # тратится на цикл, который ничего не прочитал.
                if channels_unreadable:
                    # Отдельные ошибки чтения в личку не шлём (шумно), но полное
                    # отсутствие чтения — другое дело: бот молчит целиком, и без лога
                    # об этом не узнать. Одно сообщение на инцидент, не каждый цикл.
                    if not unreadable_notified:
                        await send_error(
                            config,
                            http_client,
                            title="📡 Каналы не читаются",
                            details=[
                                "Ни один канал не прочитан — соединение с t.me "
                                "не установлено.",
                                "Проверьте сеть и VPN на машине с ботом. "
                                "Перезапуск не нужен.",
                                f"Следующая попытка через {config.poll_interval_minutes} мин.",
                            ],
                        )
                        unreadable_notified = True
                else:
                    if unreadable_notified:  # инцидент закончился — сообщаем один раз
                        await send_error(
                            config,
                            http_client,
                            title="✅ Чтение каналов восстановилось",
                            details=["Бот снова читает каналы."],
                        )
                        unreadable_notified = False
                    first_cycle = False

            # После первого тихого прохода дальше работаем в обычном режиме. Если каналы
            # не прочитаны, тихий проход своей работы не сделал (ничего не помечено seen)
            # — режим не снимаем, иначе в обычном режиме прилетят все старые вакансии.
            if config.first_run_silent and not channels_unreadable:
                logger.info("Первый тихий проход выполнен, переходим в обычный режим.")
                config = _clear_first_run_silent(config)
                interval = config.poll_interval_minutes * 60

            # Провайдер лежит — ждём дольше обычного, чтобы не жечь квоту впустую.
            if backoff:
                sleep_seconds = config.llm_backoff_minutes * 60
                logger.warning(
                    "LLM недоступна — пауза %d мин. перед следующей попыткой.",
                    config.llm_backoff_minutes,
                )
            else:
                sleep_seconds = interval

            # Ждём следующий цикл, но просыпаемся сразу по сигналу остановки.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_seconds)
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
