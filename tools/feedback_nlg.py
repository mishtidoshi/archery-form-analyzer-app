"""
feedback_nlg.py — a hand-built, rule-based natural-language generation (NLG) engine
for coaching feedback. Entirely our own code: no pretrained model, no external API,
no machine learning. The "intelligence" here is the hand-authored sentence grammar
and phrase banks below, selected and composed by ordinary Python — the same family of
technique that drives things like automated weather or financial report generators,
not a statistical language model.

WHY RULE-BASED, NOT A BIGGER TRAINED MODEL
    tools/coach_voice_model.py explains why a *trained* text generator is the wrong
    tool at this data scale (a single archer's coach-voice log is far too small to
    learn fluent English from). But writing fuller sentences doesn't actually need
    more training data if the sentences' grammar isn't being learned from a corpus in
    the first place — a template bank with several phrasing variants per finding type,
    composed with controlled randomization, is fluent and varied from the very first
    use, zero data required. The archer's own logged text becomes an optional
    PERSONALIZATION ingredient (via coach_voice_model.MarkovChain), spliced into
    individual sentences — it's not what makes the sentences grammatical.

GROUND-TRUTH DISCIPLINE
    Exactly like tools/coaching_feedback.py's basic template: every sentence here is
    built strictly from a finding's existing `metric` / `status` / `cue` fields.
    Nothing here invents, re-estimates, or restates a number — no finding field is
    ever formatted as a number in the output, only as the words already attached to
    it by the rule engine.

CALLER
    tools/coaching_feedback.py's synthesize_manual() is the only caller. It decides
    whether there's "enough language" to use compose_rich() at all
    (coach_voice_model.enough_data()) — below that threshold it uses its own basic
    bullet-point template instead, unchanged. See that function's docstring for the
    full dispatch logic.
"""
import random

from coach_voice_model import MarkovChain

OPENERS = [
    "Here's how the {view} view went — {arrows} shot{plural} on the books.",
    "{arrows} shot{plural} from the {view} view — here's the read.",
    "Looking at the {view} view: {arrows} shot{plural} to work with this time.",
    "Quick rundown on the {view} view, {arrows} shot{plural} in.",
]

GOOD_TEMPLATES = [
    "{metric} looked solid this session — {cue}.",
    "You're in a good spot with {metric}: {cue}.",
    "{metric} is right where it should be — keep leaning on {cue}.",
    "No complaints on {metric} today — {cue}.",
    "{metric} held up well. That's {cue} doing its job.",
]

ABOVE_TEMPLATES = [
    "{metric} ran a bit high today — {cue}.",
    "Keep an eye on {metric}; it's trending high. The fix: {cue}.",
    "{metric} crept above where you want it — {cue}.",
    "Worth a look: {metric} came in high. Think about {cue}.",
]

BELOW_TEMPLATES = [
    "{metric} ran a bit low today — {cue}.",
    "Keep an eye on {metric}; it's trending low. The fix: {cue}.",
    "{metric} came in under where you want it — {cue}.",
    "Worth a look: {metric} came in low. Think about {cue}.",
]

NO_FLAGS_LINES = [
    "Every scorable metric was in range this session — nothing flagged to work on.",
    "Clean session — nothing out of range to flag.",
    "Nothing flagged this time; every measured metric stayed in range.",
]

ECHO_TEMPLATES = [
    ' Closer to what your coach keeps saying: "{fragment}."',
    ' Echoes what you\'ve been hearing in practice: "{fragment}."',
    ' Worth remembering — your own notes put it this way: "{fragment}."',
]

# How often a given sentence gets a personalized echo spliced in — not every one, so
# it reads as occasional color, not a mechanical refrain every single line.
ECHO_RATE = 0.4
ECHO_MAX_WORDS = 10


def _echo(chain: MarkovChain, rng: random.Random) -> str:
    if rng.random() >= ECHO_RATE:
        return ""
    fragment = chain.generate(rng, max_words=ECHO_MAX_WORDS)
    return rng.choice(ECHO_TEMPLATES).format(fragment=fragment) if fragment else ""


def compose_rich(session, findings, active_focus, focus_as_of, coach_voice_text, rng=None):
    """Multi-sentence, paragraph-style write-up. `active_focus` is the already
    date-filtered list from synthesize_manual() — this function doesn't re-filter it.
    `rng` is injectable for tests; production calls leave it unseeded so each
    "Regenerate" click in the app gets genuinely different phrasing."""
    rng = rng or random.Random()
    chain = MarkovChain.from_text(coach_voice_text)

    view = session.get("view", "session")
    m = session.get("metrics", {})
    arrows = m.get("arrows_with_metrics") or m.get("arrows_shot") or "?"
    plural = "s" if arrows != 1 else ""

    good = [f for f in findings if f["status"] == "in_range"]
    flagged = [f for f in findings if f["status"] in ("below", "above")]

    lines = [f"**{view.capitalize()} view — {arrows} shot{plural}**", ""]
    lines.append(rng.choice(OPENERS).format(view=view, arrows=arrows, plural=plural))
    lines.append("")

    if good:
        lines.append("**What's working:**")
        for f in good:
            sentence = rng.choice(GOOD_TEMPLATES).format(metric=f["metric"], cue=f["cue"])
            lines.append(f"- {sentence}{_echo(chain, rng)}")
        lines.append("")

    if flagged:
        lines.append("**Focus for next session:**")
        for f in flagged:
            bank = ABOVE_TEMPLATES if f["status"] == "above" else BELOW_TEMPLATES
            sentence = rng.choice(bank).format(metric=f["metric"], cue=f["cue"])
            lines.append(f"- {sentence}{_echo(chain, rng)}")
        lines.append("")
    else:
        lines.append(rng.choice(NO_FLAGS_LINES))
        lines.append("")

    if active_focus:
        lines.append(f"**Active coaching focus (as of {focus_as_of}):**")
        for item in active_focus:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("_Generated locally from measured metrics and your own logged coaching "
                  "notes — no external AI model was used._")
    return "\n".join(lines)
