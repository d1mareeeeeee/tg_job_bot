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


def _build_message(match: dict) -> str:
    """Собирает текст уведомления: оценка + текст вакансии без изменений + ссылка.

    Обычный текст (без разметки) — чтобы вакансия отправлялась как есть. Если
    сообщение превышает лимит Telegram (4096), обрезаем хвост текста вакансии.
    """
    score = match["score"]
    vacancy_text = match.get("vacancy_text") or ""
    url = match.get("url")
    link_line = url if url else "ссылка недоступна (приватный канал)"

    head = f"Оценка: {score}/100\n\n"
    tail = f"\n\n{link_line}"

    # Обрезаем только текст вакансии, если не влезаем в лимит Telegram.
    budget = _TG_MAX_LEN - len(head) - len(tail)
    if len(vacancy_text) > budget:
        marker = "…(обрезано)"
        vacancy_text = vacancy_text[: budget - len(marker)] + marker

    return f"{head}{vacancy_text}{tail}"


async def send_match(config, match: dict, http_client: httpx.AsyncClient) -> None:
    """Отправляет уведомление о вакансии пользователю.

    :param config: конфигурация (токен бота, chat_id).
    :param match: dict с полями score/role/channel/reason/vacancy_text/url.
    :param http_client: общий httpx.AsyncClient.

    Обрабатывает 429 (retry_after) и прочие ошибки: логирует и не роняет цикл.
    """
    url = f"https://api.telegram.org/bot{config.notify_bot_token}/sendMessage"
    payload = {
        "chat_id": config.notify_chat_id,
        "text": _build_message(match),
        "disable_web_page_preview": False,
    }

    try:
        resp = await http_client.post(url, json=payload, timeout=30.0)
    except httpx.HTTPError as exc:
        logger.error("Сетевая ошибка при отправке уведомления: %s", exc)
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
                logger.error("Повторная отправка не удалась: %s %s", retry.status_code, retry.text)
        except httpx.HTTPError as exc:
            logger.error("Сетевая ошибка при повторной отправке: %s", exc)
        return

    logger.error("Bot API вернул ошибку %s: %s", resp.status_code, resp.text)


def _extract_retry_after(resp: httpx.Response) -> float:
    """Достаёт retry_after из тела ответа Telegram (по возможности)."""
    try:
        data = resp.json()
        return float(data.get("parameters", {}).get("retry_after", 3))
    except Exception:  # noqa: BLE001
        return 3.0
