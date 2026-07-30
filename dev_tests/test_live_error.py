"""Живая проверка: реально ли доходят сообщения об ошибках в Telegram.

Берёт настоящие NOTIFY_BOT_TOKEN/NOTIFY_CHAT_ID из .env, но LLM — всегда заглушка,
так что квота провайдера не тратится. Боевая БД не трогается: каждый сценарий
работает на временном файле.

ВНИМАНИЕ: скрипт шлёт РЕАЛЬНЫЕ сообщения вам в личку. Без флагов не делает ничего.

    python dev_tests/test_live_error.py --call    # 3 сообщения
    python dev_tests/test_live_error.py --parse   # 1 сообщение
    python dev_tests/test_live_error.py --crash   # 2 сообщения
    python dev_tests/test_live_error.py --all     # все три сценария, 6 сообщений
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from dataclasses import replace

import httpx

# На Windows при перенаправлении вывода в файл/пайп stdout получает cp1251
# и печать эмодзи из сообщений об ошибках падает с UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Скрипт лежит в dev_tests/, а модули проекта и .env — в корне.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import db
import main as m
from config import load_config
from llm_client import LLMCallError, LLMParseError

CHANNEL = "testchannel"
VACANCY = (
    "Вакансия: ML Engineer (проверка уведомлений об ошибках). Удалённо, senior. "
    "Требования: Python, PyTorch. Зарплата обсуждается."
)
TEST_MARK = "[ТЕСТ, не настоящая ошибка]"


class StubReader:
    """Отдаёт фиксированный набор постов, в сеть не ходит (каналы «прочитаны»)."""

    def __init__(self, posts):
        self._posts = posts

    async def fetch_new_posts(self, channels, lookback, is_seen):
        return list(self._posts), False


class RaisingLLM:
    """Всегда бросает заданное исключение вместо обращения к провайдеру."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def score_vacancy(self, resume_text, vacancy_text):
        raise self._exc


def make_posts(n: int, first_id: int = 900):
    return [
        {
            "channel": CHANNEL,
            "msg_id": first_id + i,
            "text": VACANCY,
            "url": f"https://t.me/{CHANNEL}/{first_id + i}",
        }
        for i in range(n)
    ]


def temp_config(**overrides):
    """Конфиг из .env, но с базой во временном каталоге — боевую не трогаем."""
    config = load_config()
    return replace(
        config,
        db_path=os.path.join(tempfile.mkdtemp(), "live.sqlite3"),
        first_run_silent=False,
        **overrides,
    )


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'OK  ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return condition


async def scenario_call(http_client: httpx.AsyncClient) -> bool:
    """LLMCallError: провайдер не отвечает. Ожидаем 3 сообщения с эскалацией."""
    print("\n=== Сценарий 1: сбой вызова LLM (ожидается 3 сообщения) ===")
    config = temp_config()
    db.init_db(config.db_path)
    try:
        exc = LLMCallError(
            "Error code: 429 - {'error': {'message': 'Provider returned error', "
            f"'code': 429}}}} {TEST_MARK}"
        )
        print(f"Порог серии: {config.llm_failure_streak}, пауза: {config.llm_backoff_minutes} мин")
        result = await m.run_cycle(
            config, StubReader(make_posts(3)), RaisingLLM(exc), http_client, "резюме кандидата"
        )
        seen = db._require_conn().execute("SELECT COUNT(*) FROM seen_posts;").fetchone()[0]
        ok = check("запрошен backoff (провайдер лежит)", result.backoff is True)
        ok &= check("посты НЕ потеряны (seen=0)", seen == 0, f"seen={seen}")
        return ok
    finally:
        db.close_db()


async def scenario_parse(http_client: httpx.AsyncClient) -> bool:
    """LLMParseError: ответ не разобран. Вакансия уходит с пометкой мимо порога."""
    print("\n=== Сценарий 2: ошибка парсинга ответа (ожидается 1 сообщение) ===")
    # Порог заведомо недостижим — убеждаемся, что пометка идёт именно мимо него.
    config = temp_config(match_threshold=100)
    db.init_db(config.db_path)
    try:
        exc = LLMParseError(f"не найден JSON в ответе модели {TEST_MARK}")
        result = await m.run_cycle(
            config, StubReader(make_posts(1)), RaisingLLM(exc), http_client, "резюме кандидата"
        )
        conn = db._require_conn()
        seen = conn.execute("SELECT COUNT(*) FROM seen_posts;").fetchone()[0]
        rows = conn.execute("SELECT score, reason FROM matches;").fetchall()
        ok = check("backoff не нужен (провайдер жив)", result.backoff is False)
        ok &= check("пост закрыт как обработанный (seen=1)", seen == 1, f"seen={seen}")
        ok &= check("в matches записан parse_error", rows == [(0, "parse_error")], str(rows))
        return ok
    finally:
        db.close_db()


async def scenario_crash() -> bool:
    """Непредвиденное падение цикла. Ожидаем 2 сообщения: дедупликация схлопывает
    три одинаковых падения в одно, четвёртое (другая ошибка) уходит отдельно.

    Гоняем НАСТОЯЩИЙ main_loop с подменённым run_cycle; пятый вызов бросает
    CancelledError — это BaseException, main_loop его не ловит и выходит через finally.
    """
    print("\n=== Сценарий 3: падение цикла + дедупликация (ожидается 2 сообщения) ===")
    print("Интервал ускорен до 0 — поэтому в сообщениях будет «через 0 мин»,")
    print("в боевом режиме там стоит реальный POLL_INTERVAL_MINUTES.")
    same = f"Не удалось открыть базу данных {TEST_MARK}"
    other = f"Неожиданный ответ Telegram API {TEST_MARK}"
    crash_script = [
        RuntimeError(same),
        RuntimeError(same),
        RuntimeError(same),
        ValueError(other),
        asyncio.CancelledError(),
    ]
    calls = {"n": 0}

    async def failing_cycle(*args, **kwargs):
        exc = crash_script[calls["n"]] if calls["n"] < len(crash_script) else asyncio.CancelledError()
        calls["n"] += 1
        raise exc

    # poll_interval_minutes=0 — цикл крутится без ожидания между итерациями.
    config = temp_config(poll_interval_minutes=0)
    real_run_cycle = m.run_cycle
    m.run_cycle = failing_cycle
    try:
        await m.main_loop(config)
    except asyncio.CancelledError:
        pass  # штатный выход из сценария
    finally:
        m.run_cycle = real_run_cycle

    return check("цикл пережил 4 падения и не умер", calls["n"] == 5, f"вызовов={calls['n']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Живая проверка доставки ошибок в Telegram. Шлёт РЕАЛЬНЫЕ сообщения.",
    )
    parser.add_argument("--call", action="store_true", help="сбой вызова LLM (3 сообщения)")
    parser.add_argument("--parse", action="store_true", help="ошибка парсинга ответа (1 сообщение)")
    parser.add_argument("--crash", action="store_true", help="падение цикла (2 сообщения)")
    parser.add_argument("--all", action="store_true", help="все три сценария (6 сообщений)")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    run_call = args.call or args.all
    run_parse = args.parse or args.all
    run_crash = args.crash or args.all

    if not (run_call or run_parse or run_crash):
        print(__doc__)
        print("Ни один сценарий не выбран — ничего не отправлено.")
        return 0

    total = 3 * run_call + 1 * run_parse + 2 * run_crash
    print(f"Будет отправлено реальных сообщений: {total}\n")

    ok = True
    http_client = httpx.AsyncClient()
    try:
        if run_call:
            ok &= await scenario_call(http_client)
        if run_parse:
            ok &= await scenario_parse(http_client)
    finally:
        await http_client.aclose()

    # main_loop заводит собственный http-клиент, отдельно от нашего.
    if run_crash:
        ok &= await scenario_crash()

    print(f"\n=== ИТОГ: {'ВСЁ ПРОШЛО' if ok else 'ЕСТЬ ПАДЕНИЯ'} ===")
    print(f"Проверьте личку: должно прийти {total} сообщени{'е' if total == 1 else 'й'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
