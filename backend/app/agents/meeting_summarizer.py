"""Map-reduce summarization of meeting transcripts with action-item extraction."""
import json
import re
from app.services.rag.chunker import chunk_text
from app.services.llm import get_chat_model


def _clean_json_response(text: str) -> str:
    """Strip markdown fences, optional language tags, and a leading ``json`` marker.

    LLMs sometimes wrap JSON in backticks or prepend the word ``json``. This
    function normalises the response so ``json.loads`` can reliably parse it.
    """
    t = text.strip()
    # Remove surrounding triple backticks with optional language specifier
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    # Remove a leading "json" marker (case‑insensitive)
    t = re.sub(r"(?i)^json\s*", "", t).strip()
    return t


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
    cleaned = _clean_json_response(resp.content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": resp.content, "decisions": [], "action_items": []}
