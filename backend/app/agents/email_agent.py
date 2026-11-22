"""Email Assistant — drafting and replying with tone control.

Provides utilities to generate email subject and body from LLM output, handling
variations such as extra blank lines, missing "Subject:" prefix, or leading
markdown fences.
"""
from app.services.llm import complete
from typing import Tuple


def _extract_subject_body(content: str) -> Tuple[str, str]:
    """Return a (subject, body) tuple from raw LLM output.

    The LLM is instructed to start with a line like ``Subject: <text>`` but in
    practice it may include extra whitespace, markdown fences, or omit the
    prefix. This helper makes the extraction tolerant:

    * Strips surrounding backticks and optional ``json`` marker.
    * Finds the first line that starts with ``Subject:`` (case‑insensitive).
    * Everything after that line is considered the body.
    """
    # Remove surrounding markdown fences and optional ``json`` marker
    cleaned = content.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    lines = cleaned.splitlines()
    subject = ""
    body_lines: list[str] = []
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            body_lines = lines[i + 1 :]
            break
    else:
        # No explicit subject line – treat the whole text as body
        body_lines = lines
    body = "\n".join(body_lines).strip()
    return subject, body


async def draft_email(intent: str, context: str, tone: str) -> dict:
    """Generate a draft email given an intent, optional context and tone.

    Returns a dictionary with ``subject`` and ``body`` keys.
    """
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    subject, body = _extract_subject_body(content)
    return {"subject": subject, "body": body}


async def reply_email(thread: str, instruction: str) -> dict:
    """Compose a reply to an existing email thread.

    The LLM is asked to start with ``Subject: Re: ...``. The helper above
    extracts the subject and body safely.
    """
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    subject, body = _extract_subject_body(content)
    return {"subject": subject, "body": body}
