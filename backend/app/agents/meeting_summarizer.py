"""Map-reduce summarization of meeting transcripts with action-item extraction.

Improves robustness of JSON parsing from LLM responses by handling optional
markdown fences and case‑insensitive ``json`` prefixes.
"""
import json
from app.services.rag.chunker import chunk_text
from app.services.llm import get_chat_model


def _clean_json_response(text: str) -> str:
    """Strip markdown fences and optional ``json`` marker from LLM output.

    The LLM may return responses wrapped in backticks or prefixed with the word
    ``json`` (in any case). This helper normalises the string so ``json.loads``
    can reliably parse it.
    """
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def summarize_meeting(transcript: str) -> dict:
    llm = get_chat_model(temperature=0.2)
    chunks = chunk_text(transcript) or [transcript]

    partials = []
    for chunk in chunks[:20]:
        resp = llm.invoke([
            ("system", "Summarize this meeting transcript segment in 3-5 bullet points."),
            ("user", chunk),
        ])
        partials.append(resp.content)

    resp = llm.invoke([
        ("system", 'Combine the partial summaries. Reply with JSON only: '
                   '{"summary": "...", "decisions": ["..."], "action_items": [{"owner": "", "task": "", "due": ""}]}'),
        ("user", "\n\n".join(partials)),
    ])
    text = _clean_json_response(resp.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback to a minimal structure if parsing fails
        return {"summary": resp.content, "decisions": [], "action_items": []}
