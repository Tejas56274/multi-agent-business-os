"""Email Assistant — drafting and replying with tone control."""
import re
from app.services.llm import complete


def _extract_subject_and_body(content: str) -> tuple[str, str]:
    """Return a cleaned subject line and body from LLM output.

    The LLM is instructed to start with a line like ``Subject: <text>``.
    This helper tolerates variations in whitespace and case, and falls back
    to returning the first line unchanged when the prefix is missing.
    """
    lines = content.splitlines()
    if not lines:
        return "", ""
    first = lines[0].strip()
    # Match "Subject:" case‑insensitively and capture the remainder.
    match = re.match(r"(?i)subject:\s*(.*)", first)
    subject = match.group(1).strip() if match else first
    body = "\n".join(lines[1:]).strip()
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
