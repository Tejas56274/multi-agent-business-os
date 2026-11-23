"""Email Assistant — drafting and replying with tone control."""
import re
from app.services.llm import complete


def _extract_json(content: str) -> str:
    """Extract a JSON object from LLM output.

    Handles cases where the model wraps the JSON in markdown fences or adds a
    ``json`` prefix. Returns a cleaned JSON string ready for ``json.loads``.
    """
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return match.group(0).strip()
    return content.strip().strip("`").removeprefix("json").strip()


async def draft_email(intent: str, context: str, tone: str) -> dict:
    content = await complete([
        ("system",
         f"You draft business emails in a {tone} tone. "
         "Return the subject on the first line prefixed 'Subject: ', then a blank line, then the body. "
         "Be concise; no placeholders left unfilled unless unavoidable."),
        ("user", f"Goal: {intent}\n\nBackground/thread:\n{context or '(none)'}"),
    ])
    # The LLM is instructed to output plain text, but we defensively extract JSON
    # in case it deviates.
    try:
        data = json.loads(_extract_json(content))
    except Exception:
        lines = content.splitlines()
        subject = lines[0].removeprefix("Subject:").strip() if lines else ""
        body = "\n".join(lines[1:]).strip()
        return {"subject": subject, "body": body}
    return data


async def reply_email(thread: str, instruction: str) -> dict:
    content = await complete([
        ("system", "Write a reply to the email thread following the user's instruction. "
                   "First line: 'Subject: Re: ...', then blank line, then body."),
        ("user", f"Thread:\n{thread}\n\nInstruction: {instruction}"),
    ])
    try:
        data = json.loads(_extract_json(content))
    except Exception:
        lines = content.splitlines()
        return {
            "subject": lines[0].removeprefix("Subject:").strip() if lines else "",
            "body": "\n".join(lines[1:]).strip()
        }
    return data
