#!/usr/bin/env python3
"""Auto-fix a single commit message to satisfy commitlint (config-conventional +
this repo's commitlint.config.js).

Designed to be driven by `git filter-branch --msg-filter` so it rewrites EVERY
commit in a PR range in place, preserving each commit's diff and all merge
topology — no destructive squash. Reads the raw message on stdin, writes the
fixed message on stdout. It is deliberately conservative: a message that already
passes is emitted byte-for-byte unchanged, so valid history is never churned.

Rules handled (all commitlint ERROR-level, i.e. the ones that actually fail CI):
  * type-enum / type-case  — unknown or mis-cased type is remapped to an allowed,
                             lower-case type (best-effort inference from wording).
  * subject-case           — a Sentence/Start/Pascal/UPPER subject is lower-cased
                             at its first character (the standard commitlint fix).
  * subject-full-stop      — a single trailing '.' is stripped.
  * header-max-length (100)— an over-long header is trimmed at a word boundary and
                             the trimmed remainder is preserved as a body line, so
                             no information is lost.
  * non-conventional header— a header with no valid type gets one inferred and
                             prefixed.

Merge/revert commits and commits already co-authored by github-actions[bot] are
left untouched (commitlint ignores them via commitlint.config.js).

Kept in sync with commitlint.config.js: update ALLOWED_TYPES and HEADER_MAX_LEN
here whenever that config changes.
"""
import re
import sys

# Must match commitlint.config.js type-enum (config-conventional + 'deps').
ALLOWED_TYPES = {
    "feat", "fix", "docs", "refactor", "perf", "test",
    "chore", "ci", "style", "revert", "deps",
}
# config-conventional header-max-length (not overridden in commitlint.config.js).
HEADER_MAX_LEN = 100

_HEADER_RE = re.compile(r"^(?P<type>\w+)(?P<scope>\([^)]*\))?(?P<bang>!)?:\s*(?P<subject>.*)$")


def _infer_type(text: str) -> str:
    """Best-effort conventional type from free-form wording."""
    t = text.strip().lower()
    if re.match(r"^(add|implement|create|introduce|support)\b", t):
        return "feat"
    if re.match(r"^(fix|correct|resolve|patch|prevent|stop|guard)\b", t):
        return "fix"
    if re.match(r"^(document|docs?\b|update docs)", t):
        return "docs"
    if re.match(r"^refactor\b", t):
        return "refactor"
    if re.match(r"^(test|cover)\b", t):
        return "test"
    if re.match(r"^(bump|upgrade|update .*version|dependency|deps)\b", t):
        return "deps"
    if re.match(r"^(perf|optimi[sz]e|speed up)\b", t):
        return "perf"
    return "chore"


def _lower_first(s: str) -> str:
    """Lower-case the first character so the subject cannot be classified as
    sentence-/start-/pascal-/upper-case (the cases config-conventional's
    subject-case rejects).

    We deliberately lower-case even a leading acronym/PascalCase class name
    ('DeepFilterNet' -> 'deepFilterNet', 'DFA' -> 'dFA'): commitlint keys the
    check off the leading character, so this is the reliable automated fix. It
    is slightly less pretty than a human rephrase, but an auto-fixer's job is to
    make CI pass deterministically — a maintainer can always reword afterward.
    Interior words (e.g. a mid-subject 'BN'/'GPU') are untouched.
    """
    if not s:
        return s
    if s[0].isupper():
        return s[0].lower() + s[1:]
    return s


def _shorten_header(prefix: str, subject: str):
    """Trim `prefix+subject` to <= HEADER_MAX_LEN at a word boundary.

    Returns (header, overflow_or_None). Any trimmed words are returned as overflow
    so the caller can preserve them in the body.
    """
    budget = HEADER_MAX_LEN - len(prefix)
    if budget <= 0:
        # Pathological: prefix alone already too long. Hard-truncate.
        return (prefix + subject)[:HEADER_MAX_LEN], None
    if len(subject) <= budget:
        return prefix + subject, None
    words = subject.split(" ")
    kept, length = [], 0
    for w in words:
        add = (1 if kept else 0) + len(w)
        if length + add <= budget:
            kept.append(w)
            length += add
        else:
            break
    if not kept:
        # A single word longer than the budget — hard split it.
        return prefix + subject[:budget], subject[budget:]
    rest = " ".join(words[len(kept):]).strip()
    header = (prefix + " ".join(kept)).rstrip()
    overflow = ("…" + rest) if rest else None
    return header, overflow


def fix_message(msg: str) -> str:
    lines = msg.split("\n")
    if not lines:
        return msg
    header = lines[0]

    # Leave commitlint-ignored commits untouched.
    if header.startswith("Merge ") or header.startswith("Revert "):
        return msg
    if "Co-Authored-By: github-actions[bot]" in msg or "Co-authored-by: github-actions[bot]" in msg:
        return msg

    body_lines = lines[1:]
    overflow = None

    m = _HEADER_RE.match(header)
    if m:
        type_ = m.group("type")
        scope = m.group("scope") or ""
        bang = m.group("bang") or ""
        subject = m.group("subject")

        type_l = type_.lower()
        if type_l not in ALLOWED_TYPES:
            type_l = _infer_type(subject)
        scope = scope.lower()
        # Drop a scope made redundant by a type remap, e.g. build(deps) -> deps.
        if scope.strip("()") == type_l:
            scope = ""
        subject = _lower_first(subject.rstrip())
        if subject.endswith(".") and not subject.endswith("..."):
            subject = subject[:-1]

        prefix = f"{type_l}{scope}{bang}: "
        header, overflow = _shorten_header(prefix, subject)
    else:
        # Non-conventional header: infer a type and prefix it.
        subject = _lower_first(header.strip())
        prefix = f"{_infer_type(header)}: "
        header, overflow = _shorten_header(prefix, subject)

    if overflow:
        # Preserve trimmed words as a body line with a leading blank (also
        # satisfies body-leading-blank when there was no body before).
        if body_lines and body_lines[0] != "":
            body_lines = ["", overflow] + body_lines
        elif body_lines:
            body_lines = [body_lines[0], overflow] + body_lines[1:]
        else:
            body_lines = ["", overflow]

    return "\n".join([header] + body_lines)


def main() -> int:
    raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    sys.stdout.buffer.write(fix_message(raw).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
