"""Email Assistant — drafting and replying with tone control.

Provides two async helpers:
- ``draft_email`` creates a new email given an intent, optional context and a desired tone.
- ``reply_email`` composes a reply to an existing thread based on an instruction.

Both functions rely on the LLM ``complete`` helper and return a ``dict`` with
``subject`` and ``body`` keys.
"""
from __future__ import annotations

import re
from typing import Dict

from app.services.llm import complete


_SUBJECT_RE = re.compile(r"^Subject:\s*(.*)", re.IGNORECASE)


def _extract_subject(text: str) -> str:
    """Return the subject line without the leading ``Subject:`` marker.

    The LLM may return the subject line with varying capitalisation, extra
    whitespace, or even surrounding markdown fences. This helper normalises the
    output by:

    1. Stripping surrounding back‑ticks and the optional ``json`` prefix.
    2. Searching for the first line that starts with ``Subject:`` (case‑insensitive).
    3. Returning the remainder of that line, stripped of surrounding whitespace.
    """
    # Remove common markdown wrappers that LLMs sometimes add.
    cleaned = text.strip().strip("`").removeprefix("json").strip()
    for line in cleaned.splitlines():
        match = _SUBJECT_RE.match(line)
        if match:
            return match.group(1).strip()
    # Fallback – return the first line if it looks like a subject.
    return cleaned.splitlines()[0].strip() if cleaned else ""


async def draft_email(intent: str, context: str, tone: str) -> Dict[str, str]:
    """Draft a business email.

    Parameters
    ----------
    intent:
        A short description of the email's purpose (e.g., "request a meeting").
    context:
        Optional background information or previous correspondence.
    tone:
        Desired tone such as ``formal``, ``friendly`` or ``concise``.

    Returns
    -------
    dict
        ``{"subject": <subject>, "body": <body>}`` where ``subject`` is the
        extracted subject line and ``body`` is the remaining content.
    """
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    subject = _extract_subject(content)
    # The body is everything after the first line that contained the subject.
    body_parts = content.splitlines()[1:]
    body = "\n".join(body_parts).strip()
    return {"subject": subject, "body": body}


async def reply_email(thread: str, instruction: str) -> Dict[str, str]:
    """Generate a reply to an existing email thread.

    Parameters
    ----------
    thread:
        The full email thread to which we are replying.
    instruction:
        The user's instruction describing the desired reply content.

    Returns
    -------
    dict
        ``{"subject": <subject>, "body": <body>}`` with the subject line stripped
        of the ``Subject:`` prefix.
    """
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    subject = _extract_subject(content)
    body_parts = content.splitlines()[1:]
    body = "\n".join(body_parts).strip()
    return {"subject": subject, "body": body}
