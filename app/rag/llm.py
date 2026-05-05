from __future__ import annotations

from openai import OpenAI

from app.config import Settings
from app.schemas import ChatMessage, Source


SYSTEM_PROMPT = """你是一个简易 ChatGPT + RAG 助手，面向中文用户工作。

规则：
1. 优先依据“参考资料”回答；资料不足时，明确说资料中没有直接依据，再给出一般性建议。
2. 引用参考资料时使用 [1]、[2] 这样的编号，不要编造不存在的来源。
3. 回答要清晰、具体、可执行；可以用简短列表，但不要堆砌。
4. 如果用户的问题与知识库无关，可以正常回答，并说明没有使用知识库依据。
"""


class MissingOpenAIKeyError(RuntimeError):
    pass


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: OpenAI | None = None
        if settings.openai_api_key:
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            self.client = OpenAI(**kwargs)

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            raise MissingOpenAIKeyError("OPENAI_API_KEY is not configured.")

        embeddings: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return embeddings

    def answer(
        self,
        history: list[ChatMessage],
        question: str,
        sources: list[Source],
    ) -> str:
        if not self.client:
            raise MissingOpenAIKeyError("OPENAI_API_KEY is not configured.")

        response = self.client.responses.create(
            model=self.settings.openai_chat_model,
            instructions=SYSTEM_PROMPT,
            input=self._build_input(history, question, sources),
            max_output_tokens=self.settings.max_output_tokens,
        )
        return extract_response_text(response)

    def _build_input(
        self,
        history: list[ChatMessage],
        question: str,
        sources: list[Source],
    ) -> list[dict[str, str]]:
        messages = [
            {"role": message.role, "content": message.content}
            for message in history[-10:]
            if message.content.strip()
        ]

        if sources:
            context = "\n\n".join(
                f"[{index}] 来源：{source.document_name}，片段 {source.chunk_index + 1}\n{source.text}"
                for index, source in enumerate(sources, start=1)
            )
            augmented_question = (
                "参考资料：\n"
                f"{context}\n\n"
                "用户问题：\n"
                f"{question}"
            )
        else:
            augmented_question = (
                "当前没有可用的参考资料，或检索没有返回资料。\n\n"
                "用户问题：\n"
                f"{question}"
            )

        messages.append({"role": "user", "content": augmented_question})
        return messages


def extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()

