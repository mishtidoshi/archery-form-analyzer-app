"""
coach_voice_model.py — a tiny language model, trained from scratch, purely on one
archer's own logged coach cues. No pretrained weights, no external API, no network
call: this is genuinely local, on-machine text generation.

WHY AN N-GRAM MODEL, NOT A NEURAL NET
    A from-scratch *neural* language model needs orders of magnitude more text than
    any single archer's coach-voice log will ever realistically contain — the app logs
    one cue at a time, appended through the Coach Notes screen, accumulating maybe a
    few dozen to a few hundred short entries over months. Training a neural net on
    that little text produces incoherent noise, not coaching language.

    A word-level n-gram (Markov chain) model is the architecture that's actually
    appropriate at this data scale: it's trained purely on the archer's own words
    (nothing pretrained), needs no GPU, retrains in milliseconds every time a new cue
    is logged (there is no persisted model file — the corpus is small enough that
    "training" is just building a dict in memory), and — because it only ever
    recombines fragments that already appear in the source text — it can only produce
    phrases in the coach's actual vocabulary, never fabricate biomechanical claims the
    way an undertrained neural net readily would.

WHAT LIVES HERE VS tools/feedback_nlg.py
    This module is just the statistical building blocks: turning raw coach_voice.md
    text into words, and recombining those words into short fragments. It has no
    opinion about coaching feedback structure. tools/feedback_nlg.py is the engine
    that actually composes a session's feedback text — rule-based sentence templates
    that weave MarkovChain fragments from here into several sentences (not just one
    bonus line), gated by enough_data() so there's a clean, honest "not enough of the
    archer's own language yet" fallback.
"""
import random
import re

_WORD_RE = re.compile(r"[A-Za-z']+")
_MD_STRIP_RE = re.compile(r"[*_`#>-]")   # markdown syntax from coach_voice.md's bold/italic/bullets
# Structural lines the app's own template writes (see main.py's add_coach_note_entry) —
# "**Coach Name** (date)" headers and "**Category:** x" labels aren't the coach's words,
# they're the app's formatting, so they're dropped before word extraction rather than
# left to leak into generated text as fake "prose".
_HEADER_LINE_RE = re.compile(r"^\*\*.+\*\*\s*\(.+\)\s*$", re.MULTILINE)
_CATEGORY_LINE_RE = re.compile(r"^\*\*Category:\*\*.*$", re.MULTILINE)
ORDER = 2   # word-pair -> next-word


def extract_words(text: str) -> list[str]:
    """Strip app-template structure (headers, category labels, markdown syntax),
    keep the actual logged prose — cue text and context — lowercased."""
    text = _HEADER_LINE_RE.sub(" ", text or "")
    text = _CATEGORY_LINE_RE.sub(" ", text)
    cleaned = _MD_STRIP_RE.sub(" ", text)
    return _WORD_RE.findall(cleaned.lower())


def enough_data(text: str, min_words: int) -> bool:
    """The one quality gate every caller should check before trying to generate
    anything from this archer's text: below `min_words` the chain has so few
    transitions that it either degenerates to echoing one sentence verbatim or has
    nowhere to go at all. A brand-new archer (or one who hasn't logged any cues yet)
    correctly returns False indefinitely — callers must have a complete, useful
    fallback for that case, never treat it as an error."""
    return len(extract_words(text)) >= min_words


class MarkovChain:
    """Order-2 word-level Markov chain built directly from `words` — the entire
    "training" step, and it's O(n)."""

    def __init__(self, words: list[str]):
        self.words = words
        self.transitions: dict[tuple, list[str]] = {}
        for i in range(len(words) - ORDER):
            key = tuple(words[i:i + ORDER])
            self.transitions.setdefault(key, []).append(words[i + ORDER])

    @classmethod
    def from_text(cls, text: str) -> "MarkovChain":
        return cls(extract_words(text))

    def generate(self, rng: random.Random, max_words: int = 12) -> str:
        """One fragment, or '' if the chain has too few words to start from (callers
        should already have checked enough_data() first; this is just a hard guard)."""
        if len(self.words) <= ORDER:
            return ""
        start = rng.randrange(len(self.words) - ORDER)
        state = tuple(self.words[start:start + ORDER])
        out = list(state)
        for _ in range(max_words - ORDER):
            choices = self.transitions.get(state)
            if not choices:
                break
            nxt = rng.choice(choices)
            out.append(nxt)
            state = (state[1], nxt)
        return " ".join(out)
