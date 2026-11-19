"""All measurements. No AI model is used as a judge."""
import random
import re

import torch
from lingua import Language, LanguageDetectorBuilder

from . import config, model_loader

_detector = None
_EDIT = set(config.EDIT_MARKERS)
_HELD = set(config.HELDOUT_MARKERS)
_ENGLISH = set(config.ENGLISH_MARKERS)


def detect_language(text):
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(
            Language.SWEDISH, Language.ENGLISH, Language.BOKMAL, Language.NYNORSK,
            Language.DANISH, Language.GERMAN, Language.DUTCH, Language.FINNISH,
            Language.FRENCH, Language.SPANISH).build()
    found = _detector.detect_language_of(text)
    return found.name if found else "NONE"


def words_of(text):
    return re.findall(r"[^\W\d_]+", text.lower())


def repeated_trigram_rate(words):
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 0.0 if not trigrams else 1 - len(set(trigrams)) / len(trigrams)


def score_reply(reply):
    words = words_of(reply)
    edit_hits = [w for w in words if w in _EDIT]
    held_hits = [w for w in words if w not in _EDIT and (w in _HELD or "æ" in w or "ø" in w)]
    english_hits = [w for w in words if w in _ENGLISH]
    language = detect_language(reply)
    repetition = repeated_trigram_rate(words)
    return {
        # Main outcome. An empty, very short or looping Swedish reply is not a successful repair.
        "valid_swedish": (language == "SWEDISH" and len(words) >= config.MIN_WORDS
                          and repetition <= config.MAX_REPETITION),
        "language": language,
        "swedish": language == "SWEDISH",
        "n_words": len(words),
        "edit_markers": len(edit_hits),
        "heldout_markers": len(held_hits),
        "english_markers": len(english_hits),
        "repetition": repetition,
        # Main outcome: the reply is Swedish and contains no Norwegian, Danish or English marker.
        "clean_swedish": language == "SWEDISH" and not (edit_hits or held_hits or english_hits),
        "marker_words": edit_hits + held_hits + english_hits,
    }


def summarize(rows):
    n = len(rows)
    total_words = max(1, sum(r["n_words"] for r in rows))
    return {
        "n": n,
        "valid_swedish_pct": 100 * sum(r["valid_swedish"] for r in rows) / n,
        "truncated_pct": 100 * sum(bool(r.get("truncated")) for r in rows) / n,
        "clean_swedish_pct": 100 * sum(r["clean_swedish"] for r in rows) / n,
        "swedish_pct": 100 * sum(r["swedish"] for r in rows) / n,
        "edit_markers_per_100w": 100 * sum(r["edit_markers"] for r in rows) / total_words,
        "heldout_markers_per_100w": 100 * sum(r["heldout_markers"] for r in rows) / total_words,
        "english_markers_per_100w": 100 * sum(r["english_markers"] for r in rows) / total_words,
        "repetition_pct": 100 * sum(r["repetition"] for r in rows) / n,
        "mean_words": sum(r["n_words"] for r in rows) / n,
    }


def run_generation(tok, model, prompts, swedish_instruction=False):
    rows = model_loader.generate(tok, model, model_loader.build_items(prompts, swedish_instruction))
    return [dict(row, **score_reply(row["reply"])) for row in rows]


# ----------------------------------------------------------------------------- probes
@torch.no_grad()
def sentence_logprobs(tok, model, sentences):
    totals = []
    tok.padding_side = "right"
    try:
        for start in range(0, len(sentences), 16):
            enc = tok(sentences[start:start + 16], return_tensors="pt", padding=True)
            logp = torch.log_softmax(model(**enc).logits[:, :-1], dim=-1)
            target = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            totals += (logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask).sum(1).tolist()
    finally:
        tok.padding_side = "left"
    return totals


def grammar_probe(tok, model, pairs):
    """A pair is correct when the model gives the grammatical sentence the higher probability."""
    good = sentence_logprobs(tok, model, [p["good"] for p in pairs])
    bad = sentence_logprobs(tok, model, [p["bad"] for p in pairs])
    return [g > b for g, b in zip(good, bad)]


# Version 1 read the wrong token for the answer letter. Version 2 fixed that, but showed that this
# model answers lettered multiple-choice questions by position (it scored 25% and 20% with four
# options, the guessing level). Version 3 needs no letters.
PROBE_VERSION = 3


@torch.no_grad()
def answer_choice_probe(tok, model, items, system=None):
    """The question is asked openly, with no options shown. Each of the four candidate answers is
    then scored as the model's reply (mean log-probability per token). The item is correct when
    the right answer gets the highest score."""
    correct = []
    tok.padding_side = "right"
    try:
        for item in items:
            prefix = model_loader.chat(item["question"], system)
            n_prefix = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
            enc = tok([prefix + option for option in item["options"]], return_tensors="pt", padding=True)
            logp = torch.log_softmax(model(**enc).logits[:, :-1], dim=-1)
            target = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            mask[:, :n_prefix - 1] = 0                      # score the answer tokens only
            token_scores = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask
            scores = (token_scores.sum(1) / mask.sum(1)).tolist()
            correct.append("ABCD"[scores.index(max(scores))] == item["answer"])
    finally:
        tok.padding_side = "left"
    return correct


def run_probes(tok, model, probes, system=None, grammar=True):
    """Answer-keyed questions are asked under the same system prompt as the generation test.
    grammar=False skips the grammar pairs (used when only the questions are scored again)."""
    knowledge = answer_choice_probe(tok, model, probes["knowledge"], system)
    general = answer_choice_probe(tok, model, probes["general"], system)
    result = {
        "knowledge_pct": 100 * sum(knowledge) / len(knowledge),
        "general_pct": 100 * sum(general) / len(general),
        "knowledge_correct": knowledge, "general_correct": general,
    }
    if grammar:
        pairs = grammar_probe(tok, model, probes["grammar"])
        by_type = {}
        for pair, ok in zip(probes["grammar"], pairs):
            by_type.setdefault(pair["type"], []).append(ok)
        result.update({
            "grammar_pct": 100 * sum(pairs) / len(pairs),
            "grammar_by_type_pct": {k: 100 * sum(v) / len(v) for k, v in by_type.items()},
            "grammar_correct": pairs,
        })
    return result


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values, seed=config.SEED, samples=config.BOOTSTRAP_SAMPLES):
    """95% interval for the mean of a list of numbers, in percent."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return 100 * means[int(0.025 * samples)], 100 * means[int(0.975 * samples) - 1]
