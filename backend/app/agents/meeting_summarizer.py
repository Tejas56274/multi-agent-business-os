"""Map-reduce summarization of meeting transcripts with action-item extraction."""
import json
import re
from app.services.rag.chunker import chunk_text
from app.services.llm import get_chat_model


def _extract_json(content: str) -> str:
    """Return the first JSON object found in *content*.

    LLMs often wrap JSON in markdown fences (```json ... ```) or prepend the word
    ``json``. This helper strips those wrappers and falls back to the original
    string if no JSON object can be detected.
    """
    # Try to find a JSON object surrounded by optional markdown fences
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return match.group(0).strip()
    # Fallback: remove common markdown wrappers
    cleaned = content.strip().strip("`").removeprefix("json").strip()
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
    text = _extract_json(resp.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If parsing still fails, return a best‑effort structure
        return {"summary": resp.content, "decisions": [], "action_items": []}
