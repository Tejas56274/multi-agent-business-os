"""Email Assistant — drafting and replying with tone control.

Provides two async helpers:
- ``draft_email`` creates a new email given an intent, context and desired tone.
- ``reply_email`` composes a reply to an existing thread based on an instruction.

Both functions now handle cases where the LLM response does not start with a
``Subject:`` line, avoiding ``IndexError`` or incorrect subject extraction.
"""
from __future__ import annotations

from typing import Dict

from app.services.llm import complete


async def draft_email(intent: str, context: str, tone: str) -> Dict[str, str]:
    """Generate a draft email.

    The LLM is prompted to return the subject on the first line prefixed with
    ``Subject:`` followed by a blank line and the body.  If the response does
    not follow this format, the function gracefully falls back to an empty
    subject and treats the whole response as the body.
    """
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    lines = content.splitlines()
    subject = ""
    body = ""
    if lines:
        first = lines[0]
        if first.lower().startswith("subject:"):
            # Properly formatted – extract subject and body
            subject = first.split(":", 1)[1].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            # No subject line – treat entire content as body
            body = "\n".join(lines).strip()
    return {"subject": subject, "body": body}


async def reply_email(thread: str, instruction: str) -> Dict[str, str]:
    """Generate a reply to an email thread.

    The LLM should return a ``Subject: Re: ...`` line followed by a blank line
    and the body.  As with ``draft_email``, the implementation tolerates missing
    or malformed subject lines.
    """
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    lines = content.splitlines()
    subject = ""
    body = ""
    if lines:
        first = lines[0]
        if first.lower().startswith("subject:"):
            subject = first.split(":", 1)[1].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = "\n".join(lines).strip()
    return {"subject": subject, "body": body}
