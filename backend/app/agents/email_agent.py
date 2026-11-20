"""Email Assistant — drafting and replying with tone control."""
from app.services.llm import complete
import re


def _extract_subject_and_body(content: str) -> tuple[str, str]:
    """Return a cleaned (subject, body) tuple.

    The LLM is instructed to start with a line like ``Subject: <text>``.
    In practice the prefix may be missing, have different casing, or be
    followed by extra whitespace. This helper normalises the output so
    callers always receive a tidy subject string and a body without a
    leading blank line.
    """
    lines = content.splitlines()
    if not lines:
        return "", ""
    # Remove a leading "Subject:" prefix case‑insensitively.
    subject_line = lines[0]
    subject = re.sub(r"^subject:\s*", "", subject_line, flags=re.IGNORECASE).strip()
    body = "\n".join(lines[1:]).lstrip("\n").strip()
    return subject, body


async def draft_email(intent: str, context: str, tone: str) -> dict:
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    subject, body = _extract_subject_and_body(content)
    return {"subject": subject, "body": body}


async def reply_email(thread: str, instruction: str) -> dict:
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    subject, body = _extract_subject_and_body(content)
    return {"subject": subject, "body": body}
