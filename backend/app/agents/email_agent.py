"""Email Assistant — drafting and replying with tone control.

Provides two async helpers:
- ``draft_email`` creates a new email given an intent, optional context and a desired tone.
- ``reply_email`` composes a reply to an existing thread based on an instruction.

Both functions return a ``dict`` with ``subject`` and ``body`` keys.
"""
from __future__ import annotations

from typing import Dict

from app.services.llm import complete


async def draft_email(intent: str, context: str, tone: str) -> Dict[str, str]:
    """Generate a draft email.

    Parameters
    ----------
    intent: str
        A short description of the email's purpose (e.g., "request a meeting").
    context: str
        Optional background information or previous thread content. If empty, a placeholder is used.
    tone: str
        Desired tone for the email (e.g., "formal", "friendly").

    Returns
    -------
    dict
        ``{"subject": <subject>, "body": <body>}`` where ``subject`` is the first line
        prefixed with ``Subject:`` as produced by the LLM and ``body`` is the remaining text.
    """
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    lines = content.splitlines()
    # Extract subject safely – if the LLM omits the prefix we still treat the first line as subject.
    if lines:
        first = lines[0]
        subject = first.removeprefix("Subject:").strip() if first.lower().startswith("subject:") else first.strip()
    else:
        subject = ""
    body = "\n".join(lines[1:]).strip()
    return {"subject": subject, "body": body}


async def reply_email(thread: str, instruction: str) -> Dict[str, str]:
    """Compose a reply to an existing email thread.

    Parameters
    ----------
    thread: str
        The full email thread to which we are replying.
    instruction: str
        The user's instruction describing the desired reply content.

    Returns
    -------
    dict
        ``{"subject": <subject>, "body": <body>}`` extracted from the LLM output.
    """
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    lines = content.splitlines()
    if lines:
        first = lines[0]
        subject = first.removeprefix("Subject:").strip() if first.lower().startswith("subject:") else first.strip()
    else:
        subject = ""
    body = "\n".join(lines[1:]).strip()
    return {"subject": subject, "body": body}
