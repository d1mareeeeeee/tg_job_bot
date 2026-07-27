"""Чтение постов из Telegram-каналов через веб-превью (без api-ключа).

Telegram отдаёт публичную HTML-версию канала по адресу
`https://t.me/s/<username>` — без авторизации и без api_id/api_hash. Мы её
парсим (httpx + BeautifulSoup) и извлекаем текстовые посты.

Ограничение способа: видны только **публичные** каналы с включённым веб-превью.
Приватные каналы и инвайт-ссылки (`t.me/+...`, `joinchat/...`) так прочитать
нельзя — они логируются и пропускаются.
"""

from __future__ import annotations

import asyncio
import logging
import math

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Реалистичный User-Agent — иначе t.me может отдать урезанную страницу.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Веб-превью отдаёт ~20 постов на страницу; используем для оценки числа страниц.
POSTS_PER_PAGE = 20
# Пауза между запросами страниц одного канала — вежливость к t.me.
PAGE_DELAY = 0.5


class WebReader:
    """Читает новые посты из каналов, парся публичное веб-превью t.me/s/."""

    def __init__(
        self,
        user_agent: str = "",
        timeout: float = 20.0,
    ):
        # Пустой user_agent → берём реалистичный дефолт.
        self._user_agent = user_agent or DEFAULT_USER_AGENT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Создаёт HTTP-клиент. Никакой авторизации не требуется."""
        self._client = httpx.AsyncClient(
            headers={"User-Agent": self._user_agent},
            timeout=self._timeout,
            follow_redirects=True,
        )
        logger.info("Веб-ридер готов (парсинг t.me/s/, без api-ключа).")

    async def disconnect(self) -> None:
        """Закрывает HTTP-клиент."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_new_posts(
        self,
        channels: list[str],
        lookback: int,
        is_seen,
    ) -> list[dict]:
        """Возвращает новые (не seen) текстовые посты из всех каналов.

        :param channels: список каналов (username; инвайт-ссылки не поддержаны).
        :param lookback: сколько последних постов проверять на канал.
        :param is_seen: функция is_seen(channel, msg_id) -> bool.
        :return: список dict-ов {channel, msg_id, text, url}.

        Ошибка на одном канале (нет доступа, сеть, таймаут) логируется и не
        прерывает обработку остальных.
        """
        posts: list[dict] = []
        for channel in channels:
            # Приватные каналы/инвайты веб-парсером недоступны — предупреждаем.
            if channel.startswith(("+", "joinchat/")):
                logger.warning(
                    "Канал %s пропущен: приватные каналы и инвайт-ссылки "
                    "не поддерживаются веб-парсером (нужен api-ключ).",
                    channel,
                )
                continue
            try:
                channel_posts = await self._fetch_channel(channel, lookback, is_seen)
                posts.extend(channel_posts)
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                logger.error("Сетевая ошибка при чтении канала %s: %s", channel, exc)
            except Exception as exc:  # noqa: BLE001 — один канал не должен ронять цикл
                logger.error("Не удалось прочитать канал %s: %s", channel, exc)
        return posts

    async def _fetch_channel(self, channel: str, lookback: int, is_seen) -> list[dict]:
        """Читает последние посты одного канала через t.me/s/, отсеивая seen.

        Как и раньше (Telethon `limit=lookback`), рассматриваем только `lookback`
        самых свежих постов канала и среди них возвращаем ещё не seen. Пагинация
        нужна лишь чтобы добрать до `lookback` рассмотренных (страница отдаёт ~20),
        а НЕ чтобы копать историю вглубь в поисках новых.
        """
        assert self._client is not None, "start() не вызван"

        result: list[dict] = []
        examined = 0  # сколько свежих постов уже рассмотрели (seen + новые)
        # Ограничиваем число страниц, чтобы не листать канал бесконечно.
        max_pages = math.ceil(lookback / POSTS_PER_PAGE) + 1
        before: int | None = None

        for _ in range(max_pages):
            url = f"https://t.me/s/{channel}"
            params = {"before": before} if before is not None else None
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()

            page_posts, min_id = self._parse_page(resp.text, channel)
            if not page_posts:
                # Пустая страница (нет постов / закрытое превью / конец истории).
                break

            # Идём от новых к старым; страница парсится сверху вниз (старые→новые).
            for post in reversed(page_posts):
                examined += 1
                if not is_seen(channel, post["msg_id"]):
                    result.append(post)
                if examined >= lookback:
                    break

            if examined >= lookback or min_id is None:
                break

            # Следующая страница — посты старше самого раннего на текущей.
            before = min_id
            await asyncio.sleep(PAGE_DELAY)

        logger.info("Канал %s: %d новых текстовых постов", channel, len(result))
        return result

    def _parse_page(self, html: str, channel: str) -> tuple[list[dict], int | None]:
        """Парсит одну страницу веб-превью.

        :return: (список постов сверху вниз, минимальный msg_id на странице).
            min_id нужен для пагинации через ?before=. None — если постов нет.
        """
        soup = BeautifulSoup(html, "html.parser")
        blocks = soup.select("div.tgme_widget_message")

        posts: list[dict] = []
        min_id: int | None = None

        for block in blocks:
            data_post = block.get("data-post")  # вида "username/12345"
            if not data_post or "/" not in data_post:
                continue
            try:
                msg_id = int(data_post.rsplit("/", 1)[1])
            except ValueError:
                continue

            min_id = msg_id if min_id is None else min(min_id, msg_id)

            text_el = block.select_one("div.tgme_widget_message_text")
            if text_el is None:
                continue  # чисто медийный пост без текста — пропускаем
            text = text_el.get_text(separator="\n").strip()
            if not text:
                continue

            posts.append(
                {
                    "channel": channel,
                    "msg_id": msg_id,
                    "text": text,
                    "url": f"https://t.me/{channel}/{msg_id}",
                }
            )

        return posts, min_id
