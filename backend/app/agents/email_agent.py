"""Email Assistant — drafting and replying with tone control.

Provides two async functions:
- ``draft_email`` creates a new email given an intent, context and tone.
- ``reply_email`` composes a reply to an existing thread based on an instruction.

Both functions call the LLM service and then parse the returned text.
The original implementation assumed the LLM always returned a line starting with
``Subject:`` exactly as‑written. In practice the model may add whitespace,
different capitalisation or even omit the subject line. The updated code
extracts the subject more defensively and falls back gracefully when the
expected format is not present.
"""
from __future__ import annotations

from typing import Dict

from app.services.llm import complete


def _parse_email_content(content: str) -> Dict[str, str]:
    """Parse LLM output into ``subject`` and ``body``.

    The LLM is instructed to start the response with a line like
    ``Subject: <title>`` followed by a blank line and the body. This helper
    tolerates leading/trailing whitespace, case variations and the absence of
    a subject line.
    """
    lines = content.splitlines()
    subject = ""
    body_lines: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Accept variations such as "subject:" or "  Subject:"
        if stripped.lower().startswith("subject:"):
            subject = stripped[len("subject:"):].strip()
            # The body starts after the first blank line following the subject
            body_start = i + 1
            # Skip a single blank line if present
            if body_start < len(lines) and lines[body_start].strip() == "":
                body_start += 1
            body_lines = lines[body_start:]
            break
    else:
        # No explicit subject line found – treat the whole content as body
        body_lines = lines
    body = "\n".join(body_lines).strip()
    return {"subject": subject, "body": body}


async def draft_email(intent: str, context: str, tone: str) -> dict:
    """Draft a new email.

    Parameters
    ----------
    intent: str
        The purpose of the email (e.g., "request a meeting").
    context: str
        Optional background information; ``'(none)'`` is used when empty.
    tone: str
        Desired tone such as "formal" or "friendly".
    """
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    return _parse_email_content(content)


async def reply_email(thread: str, instruction: str) -> dict:
    """Generate a reply to an existing email thread.

    Parameters
    ----------
    thread: str
        The full email thread to which we are replying.
    instruction: str
        The user's instruction describing the desired reply content.
    """
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    return _parse_email_content(content)
    }
    