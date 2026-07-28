"""Отправка уведомлений о подходящих вакансиях в личку через Bot API.

Используем отдельного бота-уведомителя (токен от @BotFather) и прямой вызов
метода sendMessage через httpx — без крупных bot-фреймворков.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

# Лимит длины одного сообщения Telegram sendMessage.
_TG_MAX_LEN = 4096


def _redact(value, token: str) -> str:
    """Прячет токен бота в тексте перед записью в лог.

    Токен вшит в URL Bot API, поэтому теоретически может просочиться в текст
    исключения или ответа сервера. На практике httpx в перехватываемых здесь
    ошибках URL не показывает, но лог часто прикладывают к issue — страхуемся.
    """
    text = str(value)
    return text.replace(token, "***") if token else text


def _build_message(match: dict) -> str:
    """Собирает текст уведомления: оценка + текст вакансии без изменений + ссылка.

    Обычный текст (без разметки) — чтобы вакансия отправлялась как есть. Если
    сообщение превышает лимит Telegram (4096), обрезаем хвост текста вакансии.

    При `parse_error=True` оценки нет (модель ответила неразбираемым JSON) —
    вместо неё в шапке пометка, а вакансия всё равно доходит до пользователя.
    """
    vacancy_text = match.get("vacancy_text") or ""
    url = match.get("url")
    link_line = url if url else "ссылка недоступна (приватный канал)"

    if match.get("parse_error"):
        head = "⚠️ ОШИБКА ПАРСИНГА — оценка не получена\n\n"
    else:
        head = f"Оценка: {match['score']}/100\n\n"
    tail = f"\n\n{link_line}"

    # Обрезаем только текст вакансии, если не влезаем в лимит Telegram.
    budget = _TG_MAX_LEN - len(head) - len(tail)
    if len(vacancy_text) > budget:
        marker = "…(обрезано)"
        vacancy_text = vacancy_text[: budget - len(marker)] + marker

    return f"{head}{vacancy_text}{tail}"


def _build_error_message(title: str, details: list[str]) -> str:
    """Собирает сообщение об ошибке: заголовок + непустые строки подробностей."""
    body = "\n".join(line for line in details if line)
    text = f"{title}\n\n{body}" if body else title

    # Текст ошибки от провайдера бывает длинным — режем по лимиту Telegram.
    if len(text) > _TG_MAX_LEN:
        marker = "…(обрезано)"
        text = text[: _TG_MAX_LEN - len(marker)] + marker
    return text


async def send_match(config, match: dict, http_client: httpx.AsyncClient) -> None:
    """Отправляет уведомление о вакансии пользователю.

    :param config: конфигурация (токен бота, chat_id).
    :param match: dict с полями score/role/channel/reason/vacancy_text/url.
    :param http_client: общий httpx.AsyncClient.
    """
    await _send_text(config, _build_message(match), http_client)


async def send_error(
    config,
    http_client: httpx.AsyncClient,
    *,
    title: str,
    details: list[str],
) -> None:
    """Сообщает пользователю об ошибке в работе бота.

    Никогда не бросает исключений: сбой при отправке сообщения о сбое не должен
    ронять цикл — только запись в лог.
    """
    try:
        await _send_text(config, _build_error_message(title, details), http_client)
    except Exception as exc:  # noqa: BLE001 — отправка отчёта об ошибке не критична
        logger.error(
            "Не удалось отправить уведомление об ошибке: %s",
            _redact(exc, config.notify_bot_token),
        )


async def _send_text(config, text: str, http_client: httpx.AsyncClient) -> None:
    """Отправляет готовый текст через Bot API.

    Обрабатывает 429 (retry_after) и прочие ошибки: логирует и не роняет цикл.
    """
    url = f"https://api.telegram.org/bot{config.notify_bot_token}/sendMessage"
    payload = {
        "chat_id": config.notify_chat_id,
        "text": text,
        "disable_web_page_preview": False,
    }

    token = config.notify_bot_token

    try:
        resp = await http_client.post(url, json=payload, timeout=30.0)
    except httpx.HTTPError as exc:
        logger.error("Сетевая ошибка при отправке уведомления: %s", _redact(exc, token))
        return

    if resp.status_code == 200:
        return

    # Telegram rate-limit: подождать retry_after секунд и повторить один раз.
    if resp.status_code == 429:
        retry_after = _extract_retry_after(resp)
        logger.warning("Bot API rate-limit, ждём %s сек. и повторяем", retry_after)
        await asyncio.sleep(retry_after)
        try:
            retry = await http_client.post(url, json=payload, timeout=30.0)
            if retry.status_code != 200:
                logger.error(
                    "Повторная отправка не удалась: %s %s",
                    retry.status_code, _redact(retry.text, token),
                )
        except httpx.HTTPError as exc:
            logger.error("Сетевая ошибка при повторной отправке: %s", _redact(exc, token))
        return

    logger.error("Bot API вернул ошибку %s: %s", resp.status_code, _redact(resp.text, token))


def _extract_retry_after(resp: httpx.Response) -> float:
    """Достаёт retry_after из тела ответа Telegram (по возможности)."""
    try:
        data = resp.json()
        return float(data.get("parameters", {}).get("retry_after", 3))
    except Exception:  # noqa: BLE001
        return 3.0
