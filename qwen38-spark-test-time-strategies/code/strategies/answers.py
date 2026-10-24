"""Generic answer parsing used for voting (never the expected answer): code blocks, the text after </think>,
the last 'Final answer:' line, and a number in it."""
import re


def final_section(text, thinking):
    if not thinking:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.split("</think>", 1)[1].strip() if "</think>" in text else None


def extract_code(text, langs):
    blocks = re.findall(r"```([A-Za-z0-9+#]*)[^\n]*\n(.*?)```", text, flags=re.S)
    tagged = [b for lang, b in blocks if lang.lower() in langs]
    if tagged:
        return max(tagged, key=len)
    return max((b for _, b in blocks), key=len) if blocks else None


def last_block(text):
    blocks = re.findall(r"```[A-Za-z0-9+#]*[^\n]*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def last_marker(text, marker):
    hits = [m.end() for m in re.finditer(re.escape(marker), text, flags=re.I)]
    if not hits:
        return None
    rest = text[hits[-1]:].strip()
    return rest.splitlines()[0].strip() if rest else ""


def parse_number(s):
    s = s.replace(",", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?", s)
    if not m:
        return None
    tok = m.group(0)
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b)
    return float(tok)
