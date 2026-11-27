"""Map-reduce summarization of meeting transcripts with action-item extraction.

Improved JSON extraction to handle various LLM output formats such as markdown fences
and case‑insensitive 'json' prefixes.
"""
import json
from app.services.rag.chunker import chunk_text
from app.services.llm import get_chat_model


def _clean_llm_json_output(raw: str) -> str:
    """Return a clean JSON string from LLM output.

    The LLM may wrap the JSON in markdown fences (```json ... ```) or prepend a
    language hint like "json" (case‑insensitive). This helper strips those
    decorations and any surrounding backticks or whitespace.
    """
    # Remove surrounding backticks and whitespace
    cleaned = raw.strip().strip("`")
    # Remove optional markdown fence start like "```json" or "```"
    if cleaned.startswith("```"):
        # Split on newline after the fence and drop the first line
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) > 1 else ""
    # Remove optional "json" prefix (case‑insensitive)
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:]
    return cleaned.strip()


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
    text = _clean_llm_json_output(resp.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return raw content as summary with empty lists for decisions and action items
        return {"summary": resp.content, "decisions": [], "action_items": []}
