"""Проверка новой логики seen/backoff в run_cycle — на заглушках.

Без сети, без LLM, без Telegram: подменяются reader, llm и send_match,
БД — временный файл. Проверяем, какие посты закрываются как обработанные.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import replace

# На Windows при перенаправлении вывода в файл/пайп stdout получает cp1251
# и печать эмодзи из сообщений об ошибках падает с UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Скрипт лежит в dev_tests/, а модули проекта и resume.txt — в корне.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import db
import main as m
from config import Config
from llm_client import LLMCallError, LLMParseError

VACANCY = (
    "Вакансия: ML Engineer в команду рекомендаций. Удалённо, senior. "
    "Требования: Python, PyTorch. Зарплата обсуждается."
)
NOT_VACANCY = "Всем привет, сегодня расскажем про новую модель и как она устроена внутри."

sent_messages: list[dict] = []
sent_errors: list[str] = []


class StubReader:
    """Отдаёт фиксированный набор постов."""

    def __init__(self, posts):
        self._posts = posts

    async def fetch_new_posts(self, channels, lookback, is_seen):
        return [p for p in self._posts if not is_seen(p["channel"], p["msg_id"])]


class StubLLM:
    """Отвечает по сценарию: список исходов на последовательные вызовы."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def score_vacancy(self, resume_text, vacancy_text):
        self.calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else self._outcomes_default()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def _outcomes_default():
        return {"score": 0, "fit": False, "role": "", "reason": ""}


async def fake_send_match(config, match, http_client):
    sent_messages.append(match)


async def fake_send_error(config, http_client, *, title, details):
    # Собираем так же, как реальный notifier._build_error_message.
    from notifier import _build_error_message

    sent_errors.append(_build_error_message(title, details))


def make_posts(n, text=VACANCY, channel="testch"):
    return [
        {"channel": channel, "msg_id": i, "text": text, "url": f"https://t.me/{channel}/{i}"}
        for i in range(1, n + 1)
    ]


def seen_ids(channel="testch"):
    conn = db._require_conn()
    rows = conn.execute(
        "SELECT msg_id FROM seen_posts WHERE channel = ? ORDER BY msg_id;", (channel,)
    ).fetchall()
    return [r[0] for r in rows]


def match_rows():
    conn = db._require_conn()
    return conn.execute("SELECT msg_id, score, reason FROM matches ORDER BY msg_id;").fetchall()


async def run(config, posts, outcomes):
    sent_messages.clear()
    sent_errors.clear()
    llm = StubLLM(outcomes)
    backoff = await m.run_cycle(config, StubReader(posts), llm, None, "резюме кандидата")
    return backoff, llm


def check(label, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"[{mark}] {label}{(' — ' + detail) if detail else ''}")
    return condition


async def main() -> int:
    m.send_match = fake_send_match  # подменяем отправку в Telegram
    m.send_error = fake_send_error

    tmp = os.path.join(tempfile.mkdtemp(), "test.sqlite3")
    db.init_db(tmp)

    base = Config(
        channels=["testch"],
        match_threshold=40,
        llm_rate_delay=0.0,
        notify_rate_delay=0.0,
        llm_failure_streak=3,
        llm_backoff_minutes=30,
    )
    ok = True

    # --- 1. Сбой вызова LLM: посты НЕ закрываются, после 3 подряд — backoff ---
    print("\n=== 1. LLMCallError x5 (порог серии 3) ===")
    posts = make_posts(5)
    backoff, llm = await run(base, posts, [LLMCallError("429")] * 5)
    ok &= check("backoff запрошен", backoff is True)
    ok &= check("прервались на 3-м сбое", llm.calls == 3, f"вызовов={llm.calls}")
    ok &= check("ни один пост не помечен seen", seen_ids() == [], f"seen={seen_ids()}")
    ok &= check("вакансий не отправлено", sent_messages == [])
    ok &= check("отправлено 3 сообщения об ошибке", len(sent_errors) == 3, f"={len(sent_errors)}")
    for i, expected in enumerate(["Попытка 1 из 3", "Попытка 2 из 3", "Попытка 3 из 3"], 0):
        if i < len(sent_errors):
            ok &= check(f"  сообщение {i + 1}: «{expected}»", expected in sent_errors[i])
    if sent_errors:
        ok &= check("в последнем — пауза 30 мин", "Пауза 30 мин" in sent_errors[-1],
                    repr(sent_errors[-1][-60:]))
        ok &= check("в сообщении есть ссылка на пост", "https://t.me/testch/" in sent_errors[0])
        ok &= check("в сообщении есть текст ошибки", "429" in sent_errors[0])
        print("\n--- как выглядит первое сообщение ---")
        print(sent_errors[0])
        print("--- как выглядит третье ---")
        print(sent_errors[-1])
        print()

    # --- 2. Те же посты в следующем цикле: LLM ожила ---
    print("\n=== 2. Повтор тех же постов после восстановления LLM ===")
    good = {"score": 90, "fit": True, "role": "ML Engineer", "reason": "подходит"}
    backoff, llm = await run(base, posts, [good] * 5)
    ok &= check("посты вернулись в обработку", llm.calls == 5, f"вызовов={llm.calls}")
    ok &= check("backoff не нужен", backoff is False)
    ok &= check("все 5 закрыты как seen", seen_ids() == [1, 2, 3, 4, 5], f"seen={seen_ids()}")
    ok &= check("отправлено 5 уведомлений", len(sent_messages) == 5, f"={len(sent_messages)}")
    ok &= check("сообщений об ошибках нет", sent_errors == [], f"={sent_errors}")

    # --- 3. Ошибка парсинга: пост закрывается и уходит с пометкой мимо порога ---
    print("\n=== 3. LLMParseError ===")
    db.mark_seen  # noqa: B018 — заметка: посты ниже новые, id с 10
    posts3 = [
        {"channel": "testch", "msg_id": 10, "text": VACANCY, "url": "https://t.me/testch/10"}
    ]
    backoff, llm = await run(base, posts3, [LLMParseError("битый JSON")])
    ok &= check("пост помечен seen", 10 in seen_ids(), f"seen={seen_ids()}")
    ok &= check("уведомление отправлено", len(sent_messages) == 1)
    if sent_messages:
        msg = sent_messages[0]
        ok &= check("флаг parse_error проставлен", msg.get("parse_error") is True, str(msg.get("parse_error")))
        ok &= check("текст вакансии сохранён", msg["vacancy_text"] == VACANCY)
    ok &= check("в matches записан parse_error",
                any(r[0] == 10 and r[2] == "parse_error" for r in match_rows()),
                str(match_rows()))

    # --- 4. Низкий скор и не-вакансия: seen ставим, не шлём ---
    print("\n=== 4. Низкий скор + не-вакансия ===")
    posts4 = [
        {"channel": "testch", "msg_id": 20, "text": VACANCY, "url": "https://t.me/testch/20"},
        {"channel": "testch", "msg_id": 21, "text": NOT_VACANCY, "url": "https://t.me/testch/21"},
    ]
    low = {"score": 5, "fit": False, "role": "QA", "reason": "не подходит"}
    backoff, llm = await run(base, posts4, [low])
    ok &= check("LLM вызвана только для вакансии", llm.calls == 1, f"вызовов={llm.calls}")
    ok &= check("оба поста закрыты", {20, 21} <= set(seen_ids()), f"seen={seen_ids()}")
    ok &= check("ничего не отправлено", sent_messages == [])

    # --- 5. Тихий первый запуск: всё seen, LLM не трогаем ---
    print("\n=== 5. --init (first_run_silent) ===")
    silent = replace(base, first_run_silent=True)
    posts5 = [
        {"channel": "testch", "msg_id": 30, "text": VACANCY, "url": "https://t.me/testch/30"}
    ]
    backoff, llm = await run(silent, posts5, [good])
    ok &= check("LLM не вызывалась", llm.calls == 0)
    ok &= check("пост помечен seen", 30 in seen_ids())
    ok &= check("ничего не отправлено", sent_messages == [])

    # --- 6. Сбои не подряд: серия сбрасывается, backoff не срабатывает ---
    print("\n=== 6. Чередование сбоев и успехов ===")
    posts6 = [
        {"channel": "testch", "msg_id": i, "text": VACANCY, "url": f"https://t.me/testch/{i}"}
        for i in (40, 41, 42, 43, 44)
    ]
    mixed = [LLMCallError("429"), good, LLMCallError("429"), good, LLMCallError("429")]
    backoff, llm = await run(base, posts6, mixed)
    ok &= check("backoff НЕ запрошен", backoff is False)
    ok &= check("обработаны все 5", llm.calls == 5, f"вызовов={llm.calls}")
    ok &= check("сбойные посты остались открытыми",
                not ({40, 42, 44} & set(seen_ids())), f"seen={seen_ids()}")
    ok &= check("успешные закрыты", {41, 43} <= set(seen_ids()), f"seen={seen_ids()}")
    # 3 сбоя, но не подряд: сообщений ровно 3 (лимит), все с «Попытка 1 из 3».
    ok &= check("не больше 3 сообщений об ошибке", len(sent_errors) == 3, f"={len(sent_errors)}")
    if len(sent_errors) == 3:
        ok &= check("серия не растёт (везде «Попытка 1 из 3»)",
                    all("Попытка 1 из 3" in e for e in sent_errors))
        ok &= check("в последнем — пометка про лог",
                    "только в лог" in sent_errors[-1], repr(sent_errors[-1][-60:]))
        ok &= check("паузы в сообщениях нет", not any("Пауза" in e for e in sent_errors))

    # --- 7. Дедупликация падений — гоняем НАСТОЯЩИЙ main_loop ---
    # run_cycle подменяем на падающий; интервал 0, чтобы цикл крутился мгновенно.
    # Пятый вызов бросает CancelledError — это BaseException, main_loop его не
    # перехватывает и корректно выходит через finally.
    print("\n=== 7. Дедупликация падений цикла (реальный main_loop) ===")
    sent_errors.clear()
    db.close_db()  # main_loop сам вызовет init_db

    crash_script = [
        RuntimeError("база залочена"),
        RuntimeError("база залочена"),
        RuntimeError("база залочена"),
        ValueError("другая ошибка"),
        asyncio.CancelledError(),
    ]
    calls = {"n": 0}

    async def failing_cycle(*args, **kwargs):
        exc = crash_script[calls["n"]] if calls["n"] < len(crash_script) else asyncio.CancelledError()
        calls["n"] += 1
        raise exc

    real_run_cycle = m.run_cycle
    m.run_cycle = failing_cycle
    loop_config = replace(
        base,
        poll_interval_minutes=0,          # интервал 0 → цикл без ожидания
        db_path=os.path.join(tempfile.mkdtemp(), "loop.sqlite3"),
        resume_path="resume.txt",
        llm_api_key="dummy",
        llm_model="dummy/model",
        notify_bot_token="dummy",
        notify_chat_id="0",
    )
    try:
        await m.main_loop(loop_config)
    except asyncio.CancelledError:
        pass  # штатный выход из теста
    finally:
        m.run_cycle = real_run_cycle

    ok &= check("цикл пережил 4 падения", calls["n"] == 5, f"вызовов run_cycle={calls['n']}")
    ok &= check("одинаковая ошибка ушла один раз, новая — отдельно",
                len(sent_errors) == 2, f"сообщений={len(sent_errors)}")
    if len(sent_errors) == 2:
        ok &= check("  первое — про залоченную базу", "база залочена" in sent_errors[0])
        ok &= check("  второе — про другую ошибку", "другая ошибка" in sent_errors[1])

    print("\n=== ИТОГ:", "ВСЁ ПРОШЛО" if ok else "ЕСТЬ ПАДЕНИЯ", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
