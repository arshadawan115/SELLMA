"""Small checks that run in seconds and need no model download: python -m pytest tests"""
import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src import config, evaluate, wsp_edit


def tiny_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=3, num_attention_heads=2))


def test_practice_and_test_prompts_do_not_overlap():
    dev = set(json.load(open("src/prompts_dev.json", encoding="utf-8")))
    test = json.load(open("src/prompts_test.json", encoding="utf-8"))
    assert len(test["swedish"]) == 100 == len(set(test["swedish"]))
    assert not dev & set(test["swedish"])


def test_measurement_words_are_separate_from_edit_words():
    used_for_edit = {w for pair in config.WORD_PAIRS + config.ENGLISH_PAIRS for w in pair}
    assert not used_for_edit & set(config.HELDOUT_MARKERS)


def test_reply_scoring():
    swedish = evaluate.score_reply("Det är en bra fråga och jag ska försöka svara så tydligt som möjligt.")
    assert swedish["swedish"] and swedish["clean_swedish"]
    english = evaluate.score_reply("Sure, here are three tips that you can use with your family.")
    assert english["language"] == "ENGLISH" and not english["clean_swedish"]
    mixed = evaluate.score_reply("Det är viktigt og jeg tycker inte att det fungerar.")
    assert mixed["edit_markers"] == 2 and not mixed["clean_swedish"]
    assert evaluate.repeated_trigram_rate("a b c a b c a b c".split()) > 0.5


def test_edit_removes_direction_and_restores_exactly():
    model = tiny_model()
    original = [wsp_edit.down_proj(model, i).detach().clone() for i in range(3)]
    direction = torch.randn(16)
    direction = direction / direction.norm()
    columns = [(0, 1), (2, 5)]
    backup, norm = wsp_edit.apply_edit(model, columns, direction, beta=1.0)
    assert norm > 0
    for layer, column in columns:
        assert abs(float(direction @ wsp_edit.down_proj(model, layer)[:, column])) < 1e-6
    untouched = wsp_edit.down_proj(model, 1)
    assert torch.equal(untouched, original[1])
    wsp_edit.restore(model, backup)
    for i in range(3):
        assert torch.equal(wsp_edit.down_proj(model, i), original[i])


def test_random_edit_matches_size():
    model = tiny_model()
    backup, norm, columns = wsp_edit.random_edit(model, [(0, 1), (0, 2), (2, 5)], seed=1, target_norm=0.5)
    assert abs(norm - 0.5) < 1e-6
    assert sorted(layer for layer, _ in columns) == [0, 0, 2]
    wsp_edit.restore(model, backup)
