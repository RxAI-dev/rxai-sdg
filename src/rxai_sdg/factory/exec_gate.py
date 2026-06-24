"""Programmatic execution & arithmetic gate (D5 / D4-numeric defect class).

An LLM judge scores *plausibility*; it does not execute code or do arithmetic, so a
whole class of defects passes it no matter how strong the judge model is:

* a code comment / inline literal asserts a numeric result the executed code does
  NOT produce (e.g. ``board_volume = (2*4*24)/1728   # ≈ 0.44`` - the code yields
  0.111, and the downstream narrative inverts);
* prose arithmetic ``A <op> B (≈|=) C`` whose ``C`` is wrong;
* code that *narrates* runtime behaviour it does not actually exhibit (the stdout
  buffering-mode demo) - not mechanically decidable, so FLAGGED for human review,
  never silently accepted.

This gate runs BEFORE the LLM judge as a HARD gate (auto-reject on A/B failure;
FLAG_FOR_HUMAN on the undecidable C case). It is deterministic. It deliberately
does NOT try to decide things it cannot: where execution cannot settle a claim it
emits ``flag_for_human`` rather than a pass.

Also here (programmatic, judge-independent): a 5-7-5 haiku syllable verifier and a
non-ASCII JSON-key check.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

#: relative tolerance for every numeric comparison in this module.
REL_TOL = 1e-3

# Blocks we must NOT execute (IO / network / nondeterministic / blocking).
_UNSAFE_RE = re.compile(
    r"\b(input\s*\(|while\s+True|requests\.|urllib|httpx|socket|open\s*\(|"
    r"subprocess|os\.system|os\.remove|shutil|threading|asyncio|"
    r"timeit|random\.|time\.sleep|sys\.argv)\b", re.I)


@dataclass
class GateFlag:
    kind: str            # exec_arithmetic_mismatch | inline_arithmetic_mismatch |
                         # flag_for_human_runtime_claim | haiku_syllables | json_nonascii_key
    severity: int        # 3 = hard reject; the human-flag is also gating (see below)
    turn_index: int
    segment: str         # "answer" | "reasoning"
    evidence: str
    human_review: bool = False   # True => not a confident reject; blocks silent accept


@dataclass
class ExecGateResult:
    hard_fails: list[GateFlag] = field(default_factory=list)
    human_flags: list[GateFlag] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # A human_flag is NOT a silent accept: it blocks the gate pending review.
        return not self.hard_fails and not self.human_flags

    def all_flags(self) -> list[GateFlag]:
        return self.hard_fails + self.human_flags


# ---------------------------------------------------------------------------
# code execution + variable/stdout capture
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.S)

_EXEC_WRAPPER = (
    "import json,io,contextlib\n"
    "_b=io.StringIO();_g={}\n"
    "try:\n"
    "    with contextlib.redirect_stdout(_b):\n"
    "        exec(compile(_SRC,'<m>','exec'),_g)\n"
    "except Exception as _e:\n"
    "    print(json.dumps({'error':type(_e).__name__+': '+str(_e)[:160]}))\n"
    "else:\n"
    "    _v={k:v for k,v in _g.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and not k.startswith('_')}\n"
    "    print(json.dumps({'vars':_v,'stdout':_b.getvalue()[:4000]}))\n"
)


def extract_python_blocks(text: str) -> list[str]:
    """Fenced blocks that parse as Python (python/py/text/untagged all considered)."""
    out: list[str] = []
    for lang, body in _FENCE_RE.findall(text or ""):
        if lang.lower() in ("json", "yaml", "yml", "html", "css", "bash", "sh",
                             "rust", "go", "javascript", "js", "ts", "sql", "diff"):
            continue
        try:
            ast.parse(body)
        except SyntaxError:
            continue
        out.append(body)
    return out


def run_block(code: str, timeout: float = 8.0) -> Optional[dict]:
    """Execute ``code`` in an isolated subprocess; return {vars, stdout} or None.

    Returns None when the block is unsafe to run or errors out (an erroring block
    is not a numeric contradiction, so it is not this gate's concern)."""
    if _UNSAFE_RE.search(code):
        return None
    src = "_SRC=" + repr(code) + "\n" + _EXEC_WRAPPER
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", src],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return None
    try:
        data = json.loads(out[-1])
    except (ValueError, TypeError):
        return None
    if "error" in data:
        return None
    return data


# ---------------------------------------------------------------------------
# A. code comment / inline-literal vs computed value
# ---------------------------------------------------------------------------

# a numeric assertion cue in a comment ("≈ 0.44", "= 1.14", "approx 21").
_ASSERT_NUM_RE = re.compile(
    r"(?:≈|≅|~=|==|=|\bapprox(?:imately)?\.?\b|\babout\b|\broughly\b)\s*"
    r"(-?\d[\d,]*\.?\d*)", re.I)
# units that mean the comment number is NOT a value-of-the-variable assertion.
_PROCESS_UNIT_RE = re.compile(
    r"\b(sec(?:ond)?s?|ms|millisec|minutes?|min|hours?|hrs?|days?|times|"
    r"iterations?|loops?|lines?|chars?|characters?|bytes?|steps?|calls?)\b", re.I)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*[^=].*?#\s*(.+?)\s*$")


def _rel_mismatch(a: float, b: float) -> bool:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom > REL_TOL


def _precision_abstol(claimed_str: str) -> float:
    """Half-unit-in-the-last-place tolerance for a stated literal, so a correctly
    ROUNDED result is not flagged (e.g. '12.7' tolerates ±0.05; '≈ 0.73' ±0.005).
    An integer literal tolerates ±0.5 (rounding-to-integer)."""
    s = claimed_str.strip().lstrip("+-").replace(",", "")
    if "." in s:
        return 0.5 * (10 ** -len(s.split(".")[1]))
    return 0.5


def _num_mismatch(computed: float, claimed_str: str) -> bool:
    """Mismatch if ``computed`` differs from the literal ``claimed_str`` beyond BOTH
    the relative tolerance and the stated-precision rounding tolerance."""
    try:
        claimed = float(claimed_str.replace(",", ""))
    except ValueError:
        return False
    abs_tol = max(_precision_abstol(claimed_str), REL_TOL * max(abs(computed), abs(claimed)))
    return abs(computed - claimed) > abs_tol


def check_code_arithmetic(text: str, turn_index: int, segment: str) -> list[GateFlag]:
    """A. For every executable block, compare each ``var = expr  # ... ≈ N`` comment
    literal against the value the code actually computes for ``var``."""
    flags: list[GateFlag] = []
    for block in extract_python_blocks(text):
        res = run_block(block)
        if not res:
            continue
        vars_ = res.get("vars") or {}
        for line in block.splitlines():
            m = _ASSIGN_RE.match(line)
            if not m:
                continue
            var, comment = m.group(1), m.group(2)
            if var not in vars_:
                continue
            for nm in _ASSERT_NUM_RE.finditer(comment):
                # skip if the asserted number is a process unit ("~5 seconds")
                tail = comment[nm.end():nm.end() + 14]
                if _PROCESS_UNIT_RE.match(tail.strip()):
                    continue
                computed = float(vars_[var])
                if _num_mismatch(computed, nm.group(1)):
                    flags.append(GateFlag(
                        "exec_arithmetic_mismatch", 3, turn_index, segment,
                        f"comment asserts {var}≈{nm.group(1)} but code computes "
                        f"{computed:g}"))
                break  # one mismatch per assignment line is enough
    return flags


# ---------------------------------------------------------------------------
# B. inline arithmetic in prose / markdown
# ---------------------------------------------------------------------------

_SAFE_ARITH_RE = re.compile(r"^[\d\s.()+\-*/×÷%^]+$")
#: a MAXIMAL "math chain": a run of arithmetic/relational chars (no commas, no
#: letters, no LaTeX braces) that starts and ends on a digit/paren and contains at
#: least one relational sign. Taking the MAXIMAL span avoids grabbing a partial
#: sub-expression ("2/7 = 3/28" out of "(3/8)*(2/7) = 6/56 = 3/28").
_CHAIN_RE = re.compile(
    r"[\d(][\d.()+\-*/×÷^%\s]*(?:=|≈|≅|~=|~)[\d.()+\-*/×÷^%=≈≅~\s]*[\d)%]")
_OP_RE = re.compile(r"[+\-*/×÷^]")
#: chars whose presence immediately beside a span means a non-ASCII operand was
#: dropped, so the captured expression is INCOMPLETE and must not be checked:
#: roots, unicode fractions, middot, and (right side) superscripts / 'e' notation.
_FRAC_ROOT = "√∛∜·¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅐⅑⅒"
_SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"
#: words that SCALE the magnitude (so a trailing one means the number is incomplete);
#: a plain unit ("cubic feet", "boards", "lb") does NOT scale and must still be checked.
_SCALE_WORDS = {"million", "billion", "trillion", "quadrillion", "thousand",
                "hundred", "dozen", "score", "lakh", "crore"}


def _span_truncated(text: str, start: int, end: int) -> bool:
    """True if a dropped operand / magnitude scaler abuts the span, so the captured
    expression is INCOMPLETE: '√2 ≈ 0.707', '¼ × 0.707', '1.30e6', '10⁶', '1.30 million'.
    A bare prose word ('So 192/1728 = 0.44') or a plain unit ('0.44 cubic feet') does
    NOT truncate the value and is still checked."""
    # left: a mid-number digit, a dropped root/fraction, or a trailing operator.
    j = start - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j >= 0 and (text[j].isdigit() or text[j] in _FRAC_ROOT or text[j] in "×÷*/+^·"):
        return True
    # right: glued scientific notation 'e6' / 'E-3'.
    if end < len(text) and text[end] in "eE" and end + 1 < len(text) \
            and (text[end + 1].isdigit() or text[end + 1] in "+-"):
        return True
    k = end
    while k < len(text) and text[k].isspace():
        k += 1
    if k < len(text):
        if text[k] in _FRAC_ROOT or text[k] in _SUPERSCRIPT:
            return True
        wm = re.match(r"[A-Za-z]+", text[k:])
        if wm and wm.group(0).lower() in _SCALE_WORDS:
            return True
    return False


def _safe_eval(expr: str) -> Optional[float]:
    raw = expr.strip()
    if not raw or not _SAFE_ARITH_RE.match(raw):
        return None
    e = raw.replace("×", "*").replace("÷", "/").replace("^", "**")
    if re.search(r"\*\*\s*\d{3,}", e):  # avoid huge exponentiation blowups
        return None
    try:
        node = ast.parse(e, mode="eval")
    except SyntaxError:
        return None
    for n in ast.walk(node):
        if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                              ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                              ast.USub, ast.UAdd, ast.FloorDiv)):
            return None
    try:
        val = eval(compile(node, "<a>", "eval"), {"__builtins__": {}}, {})
    except (ZeroDivisionError, ValueError, OverflowError, TypeError, SyntaxError):
        return None
    return float(val) if isinstance(val, (int, float)) else None


def check_inline_arithmetic(text: str, turn_index: int, segment: str) -> list[GateFlag]:
    """B. Verify ``A <op> B (=|≈) C`` (and longer chains) in prose/markdown; reject on
    mismatch beyond the stated-precision tolerance. Percent ('= 80%') is handled.
    Skips: definitions with no operator ('x = 5'), running-sum continuations that
    begin with an operator ('+4 = 21'), and anything with letters/LaTeX braces."""
    flags: list[GateFlag] = []
    for m in _CHAIN_RE.finditer(text or ""):
        whole = m.group(0).strip()
        # must contain a real arithmetic operator somewhere (else it's "4 = 21" noise)
        if not _OP_RE.search(whole):
            continue
        # skip if a dropped non-ASCII operand (√, ¼, e-notation, superscript) abuts
        # the span - the captured expression would be incomplete (FP source).
        if _span_truncated(text or "", m.start(), m.end()):
            continue
        raw_segs = [s.strip() for s in re.split(r"\s*(?:=|≈|≅|~=|~)\s*", whole)]
        if len(raw_segs) < 2 or any(not s for s in raw_segs):
            continue
        # a leading operator on the FIRST segment => running-sum continuation; skip.
        if raw_segs[0][0] in "+*/×÷":
            continue
        vals: list[tuple[float, str]] = []
        ok = True
        for s in raw_segs:
            pct = s.endswith("%")
            core = s[:-1].strip() if pct else s
            v = _safe_eval(core)
            if v is None:
                ok = False
                break
            vals.append((v / 100.0 if pct else v, s))
        if not ok or len(vals) < 2:
            continue
        for (a, _sa), (b, sb) in zip(vals, vals[1:]):
            # tolerance keyed to the RIGHT operand's stated precision (the "result")
            claimed_str = sb[:-1] if sb.endswith("%") else sb
            scale = 0.01 if sb.endswith("%") else 1.0
            tol = max(_precision_abstol(claimed_str) * scale,
                      REL_TOL * max(abs(a), abs(b)))
            if abs(a - b) > tol:
                flags.append(GateFlag(
                    "inline_arithmetic_mismatch", 3, turn_index, segment,
                    f"'{whole[:60]}' : {a:g} ≠ {b:g}"))
                break
    return flags


# ---------------------------------------------------------------------------
# C. claim-vs-behaviour for code that asserts runtime behaviour (best-effort)
# ---------------------------------------------------------------------------

# The stdout buffering-mode case: a single demo claims it shows BOTH the terminal
# (line-buffered) AND the redirected (fully-buffered) behaviour within one process
# run. Buffering mode is fixed for the process lifetime, so that is impossible - but
# simply running the snippet does NOT reveal the lie (you would need two separate
# invocations with/without redirection). Not mechanically decidable -> FLAG_FOR_HUMAN.
_TERMINAL_RE = re.compile(r"\b(line[\s-]?buffered|still .{0,30}\bterminal\b|attached to (?:a |the )?(?:tty|terminal)|is a (?:tty|terminal))\b", re.I)
_REDIRECT_RE = re.compile(r"\b(fully[\s-]?buffered|redirect\w*|to a (?:regular )?file|piped?)\b", re.I)
_DEMO_RE = re.compile(r"\b(demonstrat\w+|shows?|you'?ll see|prints? a (?:first|second) block|the (?:two|both) behaviou?rs|run (?:it|the script)|when run)\b", re.I)
_BUFFER_TOPIC_RE = re.compile(r"\b(buffer\w*|flush\w*|stdout|sys\.stdout)\b", re.I)


def check_runtime_claims(text: str, turn_index: int, segment: str) -> list[GateFlag]:
    """C. Best-effort. A code answer whose narration claims a SINGLE run demonstrates
    the terminal-vs-redirected stdout buffering difference is asserting runtime
    behaviour that cannot be settled by executing the snippet -> FLAG_FOR_HUMAN
    (never a silent pass). We do NOT claim a confident reject here."""
    t = text or ""
    # code may be fenced OR embedded (e.g. a JSON "code" field); detect either.
    has_code = bool(extract_python_blocks(t)) or bool(
        re.search(r"\bimport\s+\w|\bdef\s+\w+\s*\(|\bprint\s*\(|sys\.stdout", t))
    if not has_code:
        return []
    if not _BUFFER_TOPIC_RE.search(t):
        return []
    if _TERMINAL_RE.search(t) and _REDIRECT_RE.search(t) and _DEMO_RE.search(t):
        return [GateFlag(
            "flag_for_human_runtime_claim", 3, turn_index, segment,
            "code narrates a single-run terminal-vs-redirected buffering demo "
            "(buffering mode is fixed per process; not executable-decidable)",
            human_review=True)]
    return []


# ---------------------------------------------------------------------------
# secondary: 5-7-5 haiku syllable verifier
# ---------------------------------------------------------------------------

try:  # cmudict preferred; heuristic fallback when unavailable
    import cmudict as _cmu  # type: ignore
    _CMU = _cmu.dict()
except Exception:  # pragma: no cover - depends on env
    _CMU = {}

_VOWELS = "aeiouy"


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if _CMU and w in _CMU:
        pron = _CMU[w][0]
        return sum(1 for p in pron if p[-1].isdigit())
    # heuristic: count vowel groups, drop a silent trailing 'e', keep '-le' endings.
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ie", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def count_line_syllables(line: str) -> int:
    return sum(count_syllables(tok) for tok in re.findall(r"[A-Za-z']+", line))


def _extract_haiku_lines(text: str) -> Optional[list[str]]:
    """Find the 3 candidate haiku lines inside an answer (which may have a lead-in).

    Prefers a fenced block of 3 lines; else the first blank-line-bounded stanza of
    exactly 3 short, non-structural lines; else, if the whole answer is exactly 3
    such lines, those. Returns None if no clean 3-line stanza exists."""
    def clean(ln: str) -> str:
        return re.sub(r"^[>*\-\s]+", "", ln).strip()

    def to_lines(cand: str):
        ls = [clean(x) for x in cand.splitlines() if x.strip()]
        if len(ls) == 3:
            return ls
        # slash-separated single-line haiku ("line1 / line2 / line3")
        if len(ls) <= 1 and cand.count("/") >= 2:
            parts = [clean(p) for p in cand.split("/") if p.strip()]
            if len(parts) == 3:
                return parts
        return None

    # 1. a JSON "haiku"/"poem"/"verse" field value (may be slash-separated)
    for m in re.finditer(r'"(?:haiku|poem|verse)"\s*:\s*"([^"]+)"', text or "", re.I):
        got = to_lines(m.group(1))
        if got:
            return got
    # 2. a fenced block of 3 lines (or slash-separated)
    for _lang, body in _FENCE_RE.findall(text or ""):
        got = to_lines(body)
        if got:
            return got
    # 2b. whole answer as a slash-separated haiku
    if (text or "").count("/") >= 2:
        got = to_lines(text or "")
        if got:
            return got
    stanzas: list[list[str]] = []
    cur: list[str] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        structural = (not ln) or ln.startswith(("#", "|", "```", "---")) or len(ln) > 60
        if structural:
            if cur:
                stanzas.append(cur)
                cur = []
        else:
            cur.append(clean(raw))
    if cur:
        stanzas.append(cur)
    for st in stanzas:
        if len(st) == 3:
            return st
    return None


def verify_haiku(text: str) -> tuple[str, str]:
    """5-7-5 check. Returns (status, detail) where status is:
    'ok' | 'structural' (no clean 3-line haiku) | 'syllable' (3 lines, counts off).
    The caller decides severity: 'structural' and a GROSS (>=2 off) or cmudict-backed
    syllable miss hard-reject; a heuristic-only off-by-1 is flagged for human."""
    lines = _extract_haiku_lines(text)
    if lines is None:
        nonempty = [ln for ln in (text or "").splitlines() if ln.strip()]
        return "structural", f"no clean 3-line haiku (answer has {len(nonempty)} lines)"
    counts = [count_line_syllables(ln) for ln in lines]
    if counts == [5, 7, 5]:
        return "ok", "5-7-5"
    return "syllable", f"syllables {counts} != [5,7,5] ({lines!r})"


# ---------------------------------------------------------------------------
# secondary: non-ASCII JSON object keys
# ---------------------------------------------------------------------------

# Chars in a JSON key that are CONFUSABLE with ASCII or INVISIBLE - they look like a
# normal key but silently break key access (the defect). NOT flagged: visible,
# intentional identifiers such as Greek/math letters (η, α, λ) or CJK - those access
# fine under their own name and are legitimate.
_CONFUSABLE_CPS = {
    0x00A0, 0x202F, 0x00AD,                                   # nbsp / narrow nbsp / soft hyphen
    0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212,   # hyphens / dashes / minus (mimic '-')
    0x2018, 0x2019, 0x201C, 0x201D,                           # smart quotes
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF,   # zero-width / joiners / BOM / bidi
}


def _confusables(key: str) -> list[str]:
    out = []
    for c in key:
        o = ord(c)
        if o in _CONFUSABLE_CPS or 0xFF01 <= o <= 0xFF5E:   # + fullwidth ASCII
            out.append(c)
    return out


def check_json_keys(text: str, turn_index: int, segment: str) -> list[GateFlag]:
    """Reject a JSON object key containing a CONFUSABLE/INVISIBLE char (e.g. a U+2011
    non-breaking hyphen that looks like '-'): valid JSON, but silently breaks key
    access. Visible intentional non-ASCII identifiers (Greek/math/CJK) are allowed."""
    flags: list[GateFlag] = []
    seen: set[str] = set()
    for _lang, body in _FENCE_RE.findall(text or ""):
        _scan_keys(body, turn_index, segment, flags, seen)
    _scan_keys(text or "", turn_index, segment, flags, seen)
    return flags


def _scan_keys(s: str, ti: int, seg: str, flags: list, seen: set) -> None:
    for m in re.finditer(r'"([^"\n]*)"\s*:', s):
        key = m.group(1)
        bad = _confusables(key)
        if bad and key not in seen:
            seen.add(key)
            cps = ", ".join(f"U+{ord(c):04X}" for c in bad)
            flags.append(GateFlag("json_nonascii_key", 3, ti, seg,
                                  f"JSON key {key!r} has confusable/invisible char ({cps})"))


# ---------------------------------------------------------------------------
# degenerate repetition / runaway generation in the ANSWER body
# ---------------------------------------------------------------------------

# Block / shading glyphs used for ASCII bar charts. A runaway decode produces a
# single cell with thousands of these (or any char). The LLM judge has no
# repetition detector in the answer body, so this MUST hard-fail pre-judge.
_BLOCK_CHARS = "█▓░▒▉▊▋▌▍▎▏▔▕▖▗▘▙▚▛▜▝▞▟◼◾◻⬛⬜"
# 40+ identical block glyphs in a row (a real bar is a handful per cell; legit
# multi-bar charts have many SHORT runs, never one giant run).
_BLOCK_RUN_RE = re.compile(r"([" + _BLOCK_CHARS + r"])\1{39,}")
# 250+ identical of ANY char (legit markdown rules in this corpus top out ~182).
_CHAR_RUN_RE = re.compile(r"(.)\1{249,}", re.S)


def detect_degenerate_repetition(text: str) -> Optional[str]:
    """A runaway single-character run (the thousands-of-█ 'bar chart' loop)."""
    m = _BLOCK_RUN_RE.search(text or "")
    if m:
        return f"degenerate block-char run ({len(m.group(0))}x {m.group(1)!r})"
    m = _CHAR_RUN_RE.search(text or "")
    if m:
        ch = m.group(1)
        return f"degenerate char run ({len(m.group(0))}x {ch!r})"
    return None


def check_repetition(text: str, turn_index: int, segment: str) -> list[GateFlag]:
    ev = detect_degenerate_repetition(text)
    return [GateFlag("degenerate_repetition", 3, turn_index, segment, ev)] if ev else []


# ---------------------------------------------------------------------------
# table self-consistency: a "N - M noun = K" subtraction whose M contradicts the
# accompanying markdown table's own row count for that noun
# ---------------------------------------------------------------------------

# The kings/interregna case: prose states "23 - 3 interregna = 20 monarchs",
# implying 3 interregnum rows, but the table the answer itself prints has only 1
# such row. The judge reads the prose as plausible; only counting the table rows
# (which the answer authored) settles it. We require an internally-CONSISTENT
# subtraction (K == N-M, so it is not a range like "5-7 chars") and that the
# counted noun actually appears in the table, then reject if the printed table's
# row count for that noun != M.
_TABLE_SUB_RE = re.compile(
    r"(\d+)\s*[−–-]\s*(\d+)\s+([A-Za-z]{4,})\b[^.\n=]{0,20}=\s*(\d+)")


def parse_table_rows(text: str) -> list[str]:
    """Lower-cased data rows of any markdown table in ``text`` (pipe-delimited,
    excluding the ``|---|`` separator row)."""
    rows: list[str] = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if s.startswith("|") and s.count("|") >= 2 and not re.match(r"^\|[\s:|-]+\|$", s):
            rows.append(s.lower())
    return rows


def detect_table_count_mismatch(text: str) -> list[str]:
    """A ``N - M <noun> = K`` claim whose M disagrees with the number of table rows
    mentioning <noun>. Returns one evidence string per genuine contradiction."""
    rows = parse_table_rows(text)
    if not rows:
        return []
    out: list[str] = []
    for m in _TABLE_SUB_RE.finditer(text or ""):
        n, sub, word, k = int(m.group(1)), int(m.group(2)), m.group(3).lower(), int(m.group(4))
        if k != n - sub:                      # not a consistent subtraction (e.g. a range)
            continue
        stem = word.rstrip("s")[:5]
        if not any(stem in r for r in rows):  # noun isn't a table concept; skip
            continue
        cnt = sum(1 for r in rows if stem in r)
        if cnt != sub:
            out.append(f"'{n}-{sub} {word}={k}' but table has {cnt} '{word}' row(s), not {sub}")
    return out


def check_table_consistency(text: str, turn_index: int, segment: str) -> list[GateFlag]:
    return [GateFlag("table_count_mismatch", 3, turn_index, segment, ev)
            for ev in detect_table_count_mismatch(text)]


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _seg_text(turn: Any, kind: str) -> str:
    if isinstance(turn, dict):
        for s in turn.get("segments", []):
            if s.get("segment_type") == kind:
                return s.get("text") or ""
        return ""
    val = getattr(turn, kind, None)
    return val or ""


def _turn_index(turn: Any) -> int:
    return turn.get("turn_index", 0) if isinstance(turn, dict) else getattr(turn, "turn_index", 0)


def _constraint(turn: Any) -> dict:
    cs = turn.get("constraint_spec") if isinstance(turn, dict) else getattr(turn, "constraint_spec", None)
    if not cs:
        return {}
    if isinstance(cs, dict):
        return cs
    return {"type": getattr(cs, "type", None), "params": getattr(cs, "params", {})}


def run_exec_gate(turns: list, run_code: bool = True) -> ExecGateResult:
    """Run the full execution/arithmetic gate over a conversation's turns."""
    res = ExecGateResult()
    for t in turns:
        ti = _turn_index(t)
        ans = _seg_text(t, "answer")
        rea = _seg_text(t, "reasoning")
        for segment, text in (("answer", ans), ("reasoning", rea)):
            if not text:
                continue
            hard: list[GateFlag] = []
            hard += check_repetition(text, ti, segment)
            hard += check_table_consistency(text, ti, segment)
            if run_code:
                hard += check_code_arithmetic(text, ti, segment)
            hard += check_inline_arithmetic(text, ti, segment)
            hard += check_json_keys(text, ti, segment)
            # haiku only on a genre=haiku turn's answer
            if segment == "answer":
                cs = _constraint(t)
                is_haiku_cs = (cs.get("params") or {}).get("genre") == "haiku" or cs.get("type") == "haiku"
                # Only enforce when the user actually ASKED for a haiku this turn: the
                # constraint_spec metadata is sometimes stale/mislabelled (the simulator
                # planned a haiku but never phrased it), and the answer correctly
                # responds to the real question - enforcing then is a false positive.
                asked = "haiku" in _seg_text(t, "query").lower()
                if is_haiku_cs and asked:
                    status, detail = verify_haiku(text)
                    if status == "structural":
                        hard.append(GateFlag("haiku_structure", 3, ti, segment, detail))
                    elif status == "syllable":
                        counts = [int(x) for x in re.findall(r"\d+", detail.split("!=")[0])]
                        gross = any(abs(c - tgt) >= 2 for c, tgt in zip(counts, [5, 7, 5]))
                        if _CMU or gross:
                            # confident: cmudict-backed, or off by >=2 (beyond heuristic noise)
                            hard.append(GateFlag("haiku_syllables", 3, ti, segment,
                                                 f"not 5-7-5: {detail}"))
                        else:
                            # heuristic-only, off by 1: cannot confidently reject -> human
                            res.human_flags.append(GateFlag(
                                "haiku_syllables", 3, ti, segment,
                                f"heuristic syllable count off by 1 (no cmudict): {detail}",
                                human_review=True))
            res.hard_fails.extend(hard)
            for f in check_runtime_claims(text, ti, segment):
                res.human_flags.append(f)
    return res


def disposition(result: ExecGateResult) -> str:
    """Per the policy: a single-turn answer-confined exec/arith mismatch -> regenerate
    that turn; a human-flag (load-bearing runtime claim) or flags spanning >=2 turns
    -> drop the whole conversation (do not patch a cascade)."""
    if result.passed:
        return "accept"
    # a load-bearing runtime-behaviour claim (the buffering case) is a cascade -> drop.
    if any(f.kind == "flag_for_human_runtime_claim" for f in result.human_flags):
        return "drop_conversation"
    # any other human flag (e.g. a heuristic off-by-1 haiku) -> route to human review.
    if result.human_flags and not result.hard_fails:
        return "flag_for_human"
    turns = {f.turn_index for f in result.hard_fails}
    segs = {f.segment for f in result.hard_fails}
    if len(turns) >= 2:
        return "drop_conversation"
    if segs == {"answer"}:
        return "regenerate_turn"
    return "drop_conversation"
