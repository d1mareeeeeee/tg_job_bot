"""Абстракция над LLM-провайдерами.

Единый интерфейс `score_vacancy(resume_text, vacancy_text) -> dict`, за которым
скрыты конкретные реализации (OpenAI, Anthropic, Gemini). Провайдер выбирается
через переменную окружения LLM_PROVIDER — к коду он не привязан.

Каждая реализация лениво импортирует свой SDK внутри __init__, поэтому в системе
достаточно установить только пакет выбранного провайдера.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Строгий промпт из ТЗ: на выходе — чистый JSON, температура низкая.
PROMPT_TEMPLATE = """Ты — карьерный ассистент. Ниже резюме кандидата и текст вакансии из Telegram-канала.
Оцени, насколько вакансия подходит кандидату, по шкале 0–100, учитывая роль, стек/навыки, уровень (junior/middle/senior), формат работы и явные требования.
ЖЁСТКОЕ ПРАВИЛО: если формат работы в вакансии — гибрид (hybrid) или очно/офис (on-site) и полная удалёнка невозможна, верни score=0 независимо от всего остального. Если удалёнка возможна или формат работы не указан — оценивай как обычно.
Верни СТРОГО JSON: {{"score": <int>, "fit": <true|false>, "role": "<краткое название роли>", "reason": "<1-2 предложения почему подходит или нет>"}}.
Резюме: {resume_text}
Вакансия: {vacancy_text}"""

# Ответ по умолчанию при ошибке разбора / вызова LLM.
_FALLBACK = {"score": 0, "fit": False, "role": "", "reason": "parse_error"}


def _parse_json(raw: str) -> dict:
    """Надёжно достаёт JSON из ответа модели.

    Модель иногда оборачивает JSON в ```json ... ``` или добавляет текст вокруг.
    Снимаем обёртку, находим первый объект {...} регэкспом и парсим.
    При неудаче возвращаем безопасный fallback (score=0) и логируем.
    """
    if not raw:
        logger.warning("LLM вернула пустой ответ")
        return dict(_FALLBACK)

    text = raw.strip()

    # Снимаем markdown-обёртку ```json ... ``` или ``` ... ```.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Находим первый JSON-объект в тексте.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("Не найден JSON в ответе LLM: %r", raw[:200])
        return dict(_FALLBACK)

    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Ошибка парсинга JSON из ответа LLM: %s | %r", exc, raw[:200])
        return dict(_FALLBACK)

    return _normalize(data)


def _normalize(data: dict) -> dict:
    """Приводит результат к ожидаемой схеме: score 0-100 int, есть все ключи."""
    result = dict(_FALLBACK)
    result.update({k: data.get(k, result[k]) for k in result})

    # score → int в диапазоне [0, 100].
    try:
        score = int(round(float(data.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    result["score"] = max(0, min(100, score))

    result["fit"] = bool(data.get("fit", result["score"] >= 50))
    result["role"] = str(data.get("role", "") or "")
    result["reason"] = str(data.get("reason", "") or "")
    return result


class BaseLLMClient(ABC):
    """Базовый класс: собирает промпт, вызывает провайдера, парсит ответ."""

    def __init__(self, model: str, api_key: str, base_url: str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def score_vacancy(self, resume_text: str, vacancy_text: str) -> dict:
        """Оценивает соответствие вакансии резюме. Возвращает нормализованный dict."""
        prompt = PROMPT_TEMPLATE.format(
            resume_text=resume_text,
            vacancy_text=vacancy_text,
        )
        try:
            raw = self._complete(prompt)
        except Exception as exc:  # noqa: BLE001 — любой сбой SDK не должен ронять цикл
            logger.error("Ошибка вызова LLM (%s): %s", self.model, exc)
            return dict(_FALLBACK)
        return _parse_json(raw)

    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Отправляет промпт провайдеру и возвращает сырой текст ответа."""


class OpenAIClient(BaseLLMClient):
    """Провайдер OpenAI (chat.completions).

    Поддерживает OpenAI-совместимые API через base_url (напр. OpenRouter):
    задайте LLM_BASE_URL=https://openrouter.ai/api/v1 и ключ OpenRouter.
    """

    def __init__(self, model: str, api_key: str, base_url: str = "", json_mode: bool = True):
        super().__init__(model, api_key, base_url)
        from openai import OpenAI  # ленивый импорт

        self._json_mode = json_mode
        # base_url=None → SDK использует стандартный эндпоинт OpenAI.
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def _complete(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        # Не все OpenAI-совместимые модели (OpenRouter) принимают response_format.
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class AnthropicClient(BaseLLMClient):
    """Провайдер Anthropic (Messages API)."""

    def __init__(self, model: str, api_key: str, base_url: str = ""):
        super().__init__(model, api_key, base_url)
        import anthropic  # ленивый импорт

        # base_url поддержан для Anthropic-совместимых прокси; пусто = дефолт.
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url or None)

    def _complete(self, prompt: str) -> str:
        # Температуру не передаём: новые модели её не принимают, а строгий
        # промпт и так задаёт детерминированный JSON-вывод.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # Собираем текст из всех text-блоков ответа.
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return "".join(parts)


class GeminiClient(BaseLLMClient):
    """Провайдер Google Gemini (google-genai)."""

    def __init__(self, model: str, api_key: str, base_url: str = ""):
        super().__init__(model, api_key, base_url)
        from google import genai  # ленивый импорт

        if base_url:
            logger.warning("LLM_BASE_URL не поддерживается провайдером gemini — игнорируется.")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _complete(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
        return resp.text or ""


# Реестр провайдеров: имя из LLM_PROVIDER → класс реализации.
_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


def verify_free_model(config) -> None:
    """Проверяет, что модель бесплатна на OpenRouter (если включён LLM_REQUIRE_FREE).

    Работает только для OpenRouter (по base_url). Для обычных OpenAI/Anthropic/
    Gemini платность через API не определить — проверка пропускается с предупреждением.

    :raises ConfigError: если модель платная или не найдена в каталоге OpenRouter.
    """
    from config import ConfigError

    if not getattr(config, "llm_require_free", False):
        return

    base_url = getattr(config, "llm_base_url", "") or ""
    if "openrouter.ai" not in base_url:
        logger.warning(
            "LLM_REQUIRE_FREE=true, но base_url не OpenRouter — проверку платности "
            "выполнить нельзя, пропускаю."
        )
        return

    import httpx

    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {config.llm_api_key}"},
            timeout=20.0,
        )
        resp.raise_for_status()
        models = resp.json()["data"]
    except Exception as exc:  # noqa: BLE001 — сбой проверки не должен ронять запуск
        logger.warning("Не удалось проверить платность модели (%s) — пропускаю проверку.", exc)
        return

    info = next((m for m in models if m.get("id") == config.llm_model), None)
    if info is None:
        raise ConfigError(
            f"Модель {config.llm_model!r} не найдена в каталоге OpenRouter. "
            f"Проверьте LLM_MODEL (нужен id вида провайдер/модель, напр. с суффиксом :free)."
        )

    pricing = info.get("pricing", {})
    prompt_price = float(pricing.get("prompt", "0") or 0)
    completion_price = float(pricing.get("completion", "0") or 0)
    if prompt_price > 0 or completion_price > 0:
        raise ConfigError(
            f"Модель {config.llm_model!r} платная "
            f"(prompt={prompt_price}, completion={completion_price}). "
            f"LLM_REQUIRE_FREE=true разрешает только бесплатные модели — выберите "
            f"вариант с суффиксом :free или отключите проверку."
        )
    logger.info("Проверка платности пройдена: модель %s бесплатная.", config.llm_model)


def make_llm_client(config) -> BaseLLMClient:
    """Фабрика: создаёт клиент нужного провайдера по конфигурации."""
    provider = config.llm_provider
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Неизвестный LLM_PROVIDER: {provider!r}. "
            f"Доступны: {', '.join(sorted(_PROVIDERS))}."
        )
    base_url = getattr(config, "llm_base_url", "")
    logger.info(
        "LLM провайдер: %s, модель: %s%s",
        provider, config.llm_model, f", base_url: {base_url}" if base_url else "",
    )

    kwargs = {"model": config.llm_model, "api_key": config.llm_api_key, "base_url": base_url}
    # json_mode (response_format) поддерживает только OpenAI-совместимый клиент.
    if cls is OpenAIClient:
        kwargs["json_mode"] = getattr(config, "llm_json_mode", True)
    return cls(**kwargs)
