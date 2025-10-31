from __future__ import annotations
from typing import List, Optional
import re

def clean_markdown(text: str) -> str:
    s = text.replace("**", "").replace("__", "").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([\.,;:])", r"\1", s)
    s = re.sub(r"([,;:])([^\s])", r"\1 \2", s)
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s.strip()

def ensure_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s if re.search(r"[.!?)]$", s) else s + "."

def split_outside_parens(text: str, sep: str) -> List[str]:
    out, buf, depth = [], "", 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return out

def parse_numbered_markdown(s: str) -> Optional[str]:
    pattern = re.compile(r"(?:(?<=\s)|^)(\d+[\.\)])\s+", flags=re.S)
    matches = list(pattern.finditer(s))
    if not matches:
        return None

    intro = s[:matches[0].start()].strip().rstrip(" :")
    items: List[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        raw = s[start:end].strip()
        if not raw:
            continue

        head, rest = None, raw
        mhead = re.match(r"(?:\*\*)?([^*:]+?)(?:\*\*)?\s*:\s*(.*)", rest, flags=re.S)
        if mhead:
            head, rest = mhead.group(1).strip(), mhead.group(2).strip()

        if "•" in rest:
            subs = [p.strip(" •-") for p in rest.split("•") if p.strip(" •-")]
        elif "*" in rest:
            subs = [p.strip(" -*") for p in re.split(r"\s*\*\s+", rest) if p.strip(" -*")]
        else:
            subs = split_outside_parens(rest, ";")
            if len(subs) <= 1 and rest.count(",") >= 2 and len(rest) < 320:
                subs = [p.strip() for p in rest.split(",") if p.strip()]

        block_lines: List[str] = []
        if head:
            block_lines.append(ensure_period(head))
        if subs:
            if len(subs) == 1 and not head:
                block_lines.append(ensure_period(subs[0]))
            else:
                for sp in subs:
                    block_lines.append(f"- {ensure_period(sp)}")
        elif not head:
            block_lines.append(ensure_period(raw))

        items.append("\n".join(block_lines))

    numbered = "\n".join(f"{idx+1}. {it}" for idx, it in enumerate(items))
    return (intro + ":\n\n" if intro else "") + numbered

def prettify_answer(ans: str) -> str:
    s = clean_markdown(ans)
    numbered = parse_numbered_markdown(s)
    if numbered:
        return numbered
    if s.startswith("* ") or " * " in s:
        parts = [p.strip(" -*") for p in re.split(r"\s*\*\s+", s) if p.strip(" -*")]
        if len(parts) > 1:
            return "\n".join(f"- {ensure_period(p)}" for p in parts)
    if "•" in s and (s.count("•") >= 2 or "\n" not in s):
        parts = [p.strip(" •-") for p in s.split("•") if p.strip(" •-")]
        if len(parts) > 1:
            return "\n".join(f"- {ensure_period(p)}" for p in parts)
    return ensure_period(s)
