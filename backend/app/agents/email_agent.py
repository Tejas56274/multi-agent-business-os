"""Email Assistant — drafting and replying with tone control.

Provides functions to draft new emails and reply to existing threads.
Improved parsing of LLM responses to reliably extract subject lines
and body content, handling variations in formatting.
"""
from __future__ import annotations

import re
from typing import Dict

from app.services.llm import complete


def _parse_email_response(content: str) -> Dict[str, str]:
    """Extract ``subject`` and ``body`` from LLM generated email text.

    The LLM is instructed to start the response with a line like
    ``Subject: <text>`` followed by a blank line and the body.  Real‑world
    outputs can deviate (extra whitespace, missing ``Subject:``, or the
    word ``Re:`` already present).  This helper normalises those cases:

    * Looks for the first line that starts with ``Subject:`` (case‑insensitive).
    * Strips the prefix and any surrounding whitespace.
    * Treats everything after the first blank line as the body.  If a
      blank line is not found, everything after the subject line becomes
      the body.
    * If no subject line is detected, returns an empty subject and the
      full content as the body.
    """
    # Normalise line endings and split
    lines = content.replace("\r\n", "\n").split("\n")
    subject_line = ""
    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*subject:\s*", line, re.IGNORECASE):
            subject_line = re.sub(r"^\s*subject:\s*", "", line, flags=re.IGNORECASE)
            body_start = i + 1
            break
    # If a blank line follows the subject, skip it
    if body_start < len(lines) and lines[body_start].strip() == "":
        body_start += 1
    body = "\n".join(lines[body_start:]).strip()
    return {"subject": subject_line.strip(), "body": body}


async def draft_email(intent: str, context: str, tone: str) -> dict:
    """Draft a business email based on an intent, optional context and tone.

    Returns a dictionary with ``subject`` and ``body`` keys.
    """
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    return _parse_email_response(content)


async def reply_email(thread: str, instruction: str) -> dict:
    """Generate a reply to an email thread following a user instruction.

    The LLM is asked to start with ``Subject: Re: ...``.  The parsing logic
    gracefully handles variations and returns a clean ``subject`` and ``body``.
    """
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    return _parse_email_response(content)
