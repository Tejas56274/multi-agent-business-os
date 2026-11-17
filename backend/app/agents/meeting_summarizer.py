"""Map-reduce summarization of meeting transcripts with action-item extraction.

Improved JSON extraction: strips markdown fences and optional language tags
(e.g., ```json) in a case‑insensitive way before parsing.
"""
import json
import re
from app.services.rag.chunker import chunk_text
from app.services.llm import get_chat_model


def _clean_json_output(text: str) -> str:
    """Remove markdown fences and leading language identifiers from LLM output.

    The LLM may wrap JSON in code fences like ```json ... ``` or just ``` ... ```.
    This helper normalises the string so ``json.loads`` can succeed reliably.
    """
    # Strip leading/trailing backticks and optional language specifier
    cleaned = re.sub(r"^\s*```\s*\w*\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    # Remove a leading "json" word (case‑insensitive) if present
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].lstrip()
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
    text = _clean_json_output(resp.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return raw content as summary with empty lists for other fields
        return {"summary": resp.content, "decisions": [], "action_items": []}
