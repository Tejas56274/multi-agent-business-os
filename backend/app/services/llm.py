"""LLM provider abstraction.

Supports OpenAI-compatible providers such as OpenAI and Groq,
as well as local OpenAI-compatible servers such as Ollama.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings


@lru_cache
def get_chat_model(temperature: float = 0.3):
    if settings.LLM_PROVIDER == "local":
        return ChatOpenAI(
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key="not-needed",
            model=settings.LOCAL_LLM_MODEL,
            temperature=temperature,
        )

    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
        temperature=temperature,
    )


@lru_cache
def get_embeddings():
    if settings.EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY
        )

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDINGS_MODEL
    )


async def stream_chat(
    messages: list[tuple[str, str]],
    temperature: float = 0.3,
):
    """Async generator of tokens for SSE streaming."""

    llm = get_chat_model(temperature)

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


async def complete(
    messages: list[tuple[str, str]],
    temperature: float = 0.3,
) -> str:
    llm = get_chat_model(temperature)

    resp = await llm.ainvoke(messages)

    return resp.content