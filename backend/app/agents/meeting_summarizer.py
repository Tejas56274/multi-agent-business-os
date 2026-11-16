"""Map‑reduce summarization of meeting transcripts with action‑item extraction.

The module uses an LLM to produce partial bullet‑point summaries for each chunk of a
transcript and then combines them into a final JSON structure.  The original implementation
relied on a fragile series of ``strip`` and ``removeprefix`` calls to clean the LLM output,
which could fail when the model returns variations such as ``JSON`` (uppercase) or extra
whitespace.  A small helper ``_clean_json_output`` now normalises the response in a
case‑insensitive way, making the function more reliable without changing its external
behaviour.
"""
import json
from app.services.rag.chunker import chunk_text
from app.services.llm import get_chat_model


def _clean_json_output(text: str) -> str:
    """Return a JSON string stripped of markdown fences and optional language tags.

    The LLM may wrap the JSON in backticks (```) and prepend a language identifier such
    as ``json`` or ``JSON``.  This helper removes those wrappers and any surrounding
    whitespace, yielding a string that can be safely passed to ``json.loads``.
    """
    # Remove surrounding backticks and whitespace
    cleaned = text.strip().strip("`")
    # Remove a leading language tag (case‑insensitive) if present
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:]
    return cleaned.strip()


def summarize_meeting(transcript: str) -> dict:
    """Summarize a meeting transcript.

    The function splits the transcript into chunks, asks the LLM to produce a short
    bullet‑point summary for each chunk, and finally asks the LLM to combine those
    partial summaries into a single JSON object with ``summary``, ``decisions`` and
    ``action_items`` fields.
    """
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
    cleaned = _clean_json_output(resp.content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return the raw content as the summary with empty lists for the other fields
        return {"summary": resp.content, "decisions": [], "action_items": []}
    
