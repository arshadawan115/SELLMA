"""Runs the whole experiment. Finished stages are read from results/ and not repeated.

    python -m src.pilot                     # 0. how often does the unchanged model fail?
    python -m src.run_experiment            # all stages below
    python -m src.run_experiment weights    # 1. weights-only edit (no forward pass)
    python -m src.run_experiment dev        # 2. three feedback rounds on the practice prompts
    python -m src.run_experiment test       # 3. all conditions on the frozen test prompts
    python -m src.run_experiment report     # 4. tables, examples and the chart

The targeted edit (round 3) in one line:
    layer L MLP output matrix, n near-constant neurons c:  W[:, c] += alpha * v / (n * m_c)
where v is the mean residual-stream difference "no system prompt minus English system prompt"
after layer L on the practice prompts, and m_c is the mean activation of neuron c.
"""
import json
import os
import sys
import time

import torch

from . import activation_analysis, config, evaluate, model_loader, steering, wsp_edit

K_WEIGHTS_ONLY, BETA_WEIGHTS_ONLY = 128, 1.0          # fixed in advance, never tuned
ROUND1_EDITS = [(k, beta) for k in (32, 128, 512) for beta in (1.0, 2.0)]
ROUND1_STEERING = [(layer, alpha) for layer in (8, 14) for alpha in (0.5, 1.0, 2.0)]
ROUND2_EDITS = [(128, 1.0), (128, 2.0)]
ROUND3_EDITS = [(alpha, n) for alpha in (1.0, 2.0) for n in (1, 16)]   # strength, neurons
RANDOM_SEEDS = (1, 2, 3, 4, 5, 6)   # 4-6 were added after seeds 1-3 disagreed; all six are reported
CONDITIONS = (["baseline", "swedish_prompt", "edit_targeted"]
              + [f"random_edit_{seed}" for seed in RANDOM_SEEDS]
              + ["steering", "edit_weights_only"])        # the last two are optional extras
N_TEST, N_NO_SYSTEM = 60, 20     # test prompts used: first 60 with the English system prompt,
                                 # first 20 also without a system prompt (fixed before any test run)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, ensure_ascii=False, indent=1)


def rounded(summary):
    return {k: round(v, 1) for k, v in summary.items()}


# ----------------------------------------------------------------------------- stage 1
def stage_weights(tok, model):
    if os.path.exists("results/weight_reading.json"):
        return
    gain = model.model.norm.weight.detach()
    direction = wsp_edit.non_swedish_direction(tok, model)
    projections = wsp_edit.column_projections(model, direction)
    columns = wsp_edit.top_columns(projections.abs(), K_WEIGHTS_ONLY)
    examples = []
    for layer, column in columns[:8]:
        vector = wsp_edit.down_proj(model, layer).detach()[:, column] * gain
        examples.append({"layer": layer, "column": column,
                         "projection": float(projections[layer, column]),
                         "raises_when_positive": wsp_edit.top_tokens(tok, model, vector, 10),
                         "raises_when_negative": wsp_edit.top_tokens(tok, model, -vector, 10)})
    write("results/weight_reading.json", {
        "forward_passes": 0,
        "pairs": config.DIRECTION_PAIRS + config.ENGLISH_PAIRS,
        "direction_raises": wsp_edit.top_tokens(tok, model, direction * gain, 30),
        "direction_lowers": wsp_edit.top_tokens(tok, model, -direction * gain, 30),
        "top_column_examples": examples,
        "edit": wsp_edit.manifest(model, "edit_weights_only",
                                  "direct weight reasoning (no forward pass)", columns,
                                  BETA_WEIGHTS_ONLY, None,
                                  {"column_rule": "largest |d . W[:, c]| over all layers"}),
    })


# ----------------------------------------------------------------------------- stage 2
def practice_rows(setting):
    return [r for r in read("results/pilot_dev.json")["rows"] if r["setting"] == setting]


def dev_activity(tok, model):
    """Activity of the unchanged model on the practice prompts: English system prompt versus no
    system prompt. Returns (neuron difference, residual difference towards 'no system', passes)."""
    neurons_en, residual_en, n1 = activation_analysis.mean_activity(tok, model, practice_rows("english_system"))
    neurons_sv, residual_sv, n2 = activation_analysis.mean_activity(tok, model, practice_rows("no_system"))
    return neurons_en - neurons_sv, residual_sv - residual_en, n1 + n2


def activity_directions(residual_shift):
    """One unit direction per layer: where the residual stream moves when the system prompt is English."""
    towards_english = -residual_shift[1:]
    return towards_english / towards_english.norm(dim=1, keepdim=True)


def dev_generate(tok, model, prompts, label, log):
    start = time.time()
    rows = model_loader.generate(tok, model, model_loader.build_items(prompts, settings=("english_system",)))
    summary = evaluate.summarize([dict(r, **evaluate.score_reply(r["reply"])) for r in rows])
    summary["seconds"] = round(time.time() - start)
    log[label] = summary
    print(label, rounded(summary), flush=True)
    return summary


def stage_round1(tok, model):
    """Round 1: projection edits along the weights-only direction, neurons chosen by activity;
    and the steering grid."""
    if os.path.exists("results/dev_round1.json"):
        return
    prompts = read("src/prompts_dev.json")
    direction = wsp_edit.non_swedish_direction(tok, model)
    neuron_shift, residual_shift, passes = dev_activity(tok, model)
    push = neuron_shift * wsp_edit.column_projections(model, direction)
    log, best_steer = {}, None
    for k, beta in ROUND1_EDITS:
        backup, _ = wsp_edit.apply_edit(model, wsp_edit.top_columns(push, k), direction, beta)
        dev_generate(tok, model, prompts, f"edit_k{k}_beta{beta}", log)
        wsp_edit.restore(model, backup)
    for layer, alpha in ROUND1_STEERING:
        with steering.Steering(model, layer, alpha * residual_shift[layer + 1]):
            summary = dev_generate(tok, model, prompts, f"steer_layer{layer}_alpha{alpha}", log)
        if best_steer is None or summary["swedish_pct"] > best_steer["swedish_pct"]:
            best_steer = dict(summary, layer=layer, alpha=alpha)
    write("results/dev_round1.json", {
        "round": 1, "activation_forward_passes": passes, "grid": log,
        "steering": {"layer": best_steer["layer"], "alpha": best_steer["alpha"],
                     "vector": "mean residual without system prompt minus mean residual with English "
                               "system prompt, after the chosen layer, on practice prompts"},
    })


def stage_round2(tok, model):
    """Round 2: the same projection edit, with the direction itself taken from activity."""
    if os.path.exists("results/dev_round2_activity_direction.json"):
        return
    prompts = read("src/prompts_dev.json")
    neuron_shift, residual_shift, _ = dev_activity(tok, model)
    directions = activity_directions(residual_shift)
    push = neuron_shift * wsp_edit.column_projections(model, directions)
    log = {}
    dev_generate(tok, model, prompts, "baseline_same_batching", log)
    for k, beta in ROUND2_EDITS:
        backup, _ = wsp_edit.apply_edit(model, wsp_edit.top_columns(push, k), directions, beta)
        dev_generate(tok, model, prompts, f"edit_k{k}_beta{beta}", log)
        wsp_edit.restore(model, backup)
    write("results/dev_round2_activity_direction.json", {"round": 2, "grid": log})


def steady_neurons(tok, model, layer, n):
    """The n neurons of one layer whose activation varies least relative to its mean."""
    mean, std = activation_analysis.neuron_stats(tok, model, read("results/pilot_dev.json")["rows"], layer)
    chosen = torch.topk(mean.abs() / (std + 1e-6), n).indices.tolist()
    return chosen, mean, {"mean": [float(mean[c]) for c in chosen], "std": [float(std[c]) for c in chosen]}


def apply_targeted(tok, model, spec, vector):
    _, mean, _ = steady_neurons(tok, model, spec["layer"], len(spec["neurons"]))
    return wsp_edit.apply_constant_push(model, spec["layer"], spec["neurons"], mean, vector)


def stage_round3(tok, model):
    """Round 3: the targeted edit. The steering push that worked in round 1 is written into the
    weights of a few near-constant neurons in the same layer. Also runs the confound check."""
    if os.path.exists("results/dev_round3_targeted.json"):
        return
    prompts = read("src/prompts_dev.json")
    layer = read("results/dev_round1.json")["steering"]["layer"]
    _, residual_shift, _ = dev_activity(tok, model)
    log, best = {}, None
    baseline = dev_generate(tok, model, prompts, "baseline", log)
    for alpha, n in ROUND3_EDITS:
        neurons, mean, stats = steady_neurons(tok, model, layer, n)
        spec = {"layer": layer, "neurons": neurons, "alpha": alpha, "neuron_stats": stats}
        backup, norm = wsp_edit.apply_constant_push(model, layer, neurons, mean,
                                                    alpha * residual_shift[layer + 1])
        summary = dev_generate(tok, model, prompts, f"push_alpha{alpha}_n{n}", log)
        wsp_edit.restore(model, backup)
        # Rule: most valid Swedish replies; on a tie the earlier (weaker, smaller) setting stays.
        if best is None or summary["valid_swedish_pct"] > best[1]["valid_swedish_pct"]:
            best = (dict(spec, change_norm=norm), summary)
    spec = best[0]
    weights = len(spec["neurons"]) * model.config.hidden_size
    spec.update({
        "name": "edit_targeted",
        "method": "iterative model-feedback optimisation: direction and neurons from activations, "
                  "three feedback rounds on practice prompts",
        "tensor": f"model.layers.{layer}.mlp.down_proj.weight[:, c]",
        "update": "W[:, c] += alpha * v / (n * m_c)",
        "weights_changed": weights,
        "share_of_model_pct": 100 * weights / sum(p.numel() for p in model.parameters()),
    })

    # Confound check: is the push about language, or about a system prompt being present?
    items = [{"setting": "swedish_system", "prompt": p, "text": model_loader.chat(p, config.SWEDISH_SYSTEM)}
             for p in prompts]
    rows = [dict(r, **evaluate.score_reply(r["reply"])) for r in model_loader.generate(tok, model, items)]
    _, residual_swedish_system, _ = activation_analysis.mean_activity(tok, model, rows)
    _, residual_english_system, _ = activation_analysis.mean_activity(tok, model, practice_rows("english_system"))
    main = residual_shift[layer + 1]
    language_only = (residual_swedish_system - residual_english_system)[layer + 1]
    confound = {
        "question": "Does the push separate English from Swedish system prompts, or only 'system "
                    "prompt present' from 'absent'?",
        "swedish_system_prompt": evaluate.summarize(rows),
        "cosine_main_vs_language_only": float(torch.nn.functional.cosine_similarity(main, language_only, dim=0)),
        "norm_main": float(main.norm()), "norm_language_only": float(language_only.norm()),
    }
    print("confound", confound["cosine_main_vs_language_only"], rounded(confound["swedish_system_prompt"]), flush=True)
    write("results/dev_round3_targeted.json", {
        "round": 3, "rule": "most valid Swedish replies on the 30 practice prompts with the English "
                            "system prompt; on a tie the weaker and smaller setting is kept",
        "baseline": baseline, "grid": log, "edit": spec, "confound_check": confound,
    })


# ----------------------------------------------------------------------------- stage 3
def measure(tok, model, test, probes, swedish_prompt=False):
    start = time.time()
    english_setting = model_loader.build_items(test["swedish"][:N_TEST], swedish_prompt, ("english_system",))
    no_system = model_loader.build_items(test["swedish"][:N_NO_SYSTEM], swedish_prompt, ("no_system",))
    english_users = model_loader.build_items(test["english"], swedish_prompt, ("english_system",))
    rows = [dict(r, **evaluate.score_reply(r["reply"]))
            for r in model_loader.generate(tok, model, english_setting + no_system + english_users)]
    a, b = len(english_setting), len(english_setting) + len(no_system)
    system = config.ENGLISH_SYSTEM + (" " + config.SWEDISH_INSTRUCTION if swedish_prompt else "")
    result = {
        "english_system": evaluate.summarize(rows[:a]),
        "no_system": evaluate.summarize(rows[a:b]),
        "english_users_answered_in_english_pct":
            100 * sum(r["language"] == "ENGLISH" for r in rows[b:]) / len(rows[b:]),
        "rows_english_system": rows[:a], "rows_no_system": rows[a:b], "rows_english_users": rows[b:],
    }
    result.update(evaluate.run_probes(tok, model, probes, system))
    result["seconds"] = round(time.time() - start)
    return result


def stage_test(tok, model):
    test, probes = read("src/prompts_test.json"), read("src/probes.json")
    weights_only = read("results/weight_reading.json")["edit"]
    steer = read("results/dev_round1.json")["steering"]
    targeted = read("results/dev_round3_targeted.json")["edit"]
    direction = wsp_edit.non_swedish_direction(tok, model)
    _, residual_shift, _ = dev_activity(tok, model)
    push = targeted["alpha"] * residual_shift[targeted["layer"] + 1]
    manifest = {"model": config.MODEL, "revision": config.REVISION, "edits": []}

    def fingerprint():
        return float(sum(wsp_edit.down_proj(model, i).double().sum()
                         for i in range(model.config.num_hidden_layers)))

    before = fingerprint()
    for name in CONDITIONS:
        path = f"results/test/{name}.json"
        backup, info = None, None
        if name == "edit_targeted":
            backup, norm = apply_targeted(tok, model, targeted, push)
            info = dict(targeted, change_norm=norm)
        elif name.startswith("random_edit"):
            # Same layer, same neurons, same form and same size as the targeted edit; random direction.
            seed = int(name.rsplit("_", 1)[1])
            generator = torch.Generator().manual_seed(seed)
            random_vector = torch.randn(model.config.hidden_size, generator=generator)
            random_vector = random_vector / random_vector.norm() * push.norm()
            backup, norm = apply_targeted(tok, model, targeted, random_vector)
            info = {"name": name, "method": "random control", "seed": seed, "layer": targeted["layer"],
                    "neurons": targeted["neurons"], "update": "W[:, c] += r / (n * m_c), |r| = |alpha * v|",
                    "weights_changed": targeted["weights_changed"], "change_norm": norm}
        elif name == "edit_weights_only":
            backup, norm = wsp_edit.apply_edit(model, [tuple(c) for c in weights_only["columns"]],
                                               direction, weights_only["beta"])
            info = dict(weights_only, change_norm=norm)
        if info:
            manifest["edits"].append(info)
        swedish_prompt = name == "swedish_prompt"
        system = config.ENGLISH_SYSTEM + (" " + config.SWEDISH_INSTRUCTION if swedish_prompt else "")
        steer_vector = steer["alpha"] * residual_shift[steer["layer"] + 1]

        def run(job):
            if name == "steering":
                with steering.Steering(model, steer["layer"], steer_vector):
                    return job()
            return job()

        if not os.path.exists(path):
            result = run(lambda: measure(tok, model, test, probes, swedish_prompt))
            result["probe_version"] = evaluate.PROBE_VERSION
            write(path, dict(condition=name, **result))
        elif read(path).get("probe_version") != evaluate.PROBE_VERSION:
            # Replies are kept; only the answer-keyed questions are scored again.
            result = read(path)
            result.update(run(lambda: evaluate.run_probes(tok, model, probes, system, grammar=False)))
            result["probe_version"] = evaluate.PROBE_VERSION
            write(path, result)
        else:
            result = None
        if result:
            print(name, result["seconds"], "s | valid Swedish", round(result["english_system"]["valid_swedish_pct"], 1),
                  "| Swedish", round(result["english_system"]["swedish_pct"], 1),
                  "| words", round(result["english_system"]["mean_words"], 1),
                  "| no-system", round(result["no_system"]["swedish_pct"], 1),
                  "| knowledge", round(result["knowledge_pct"], 1),
                  "| general", round(result["general_pct"], 1),
                  "| grammar", round(result["grammar_pct"], 1),
                  "| English users", round(result["english_users_answered_in_english_pct"], 1), flush=True)
        if backup is not None:
            wsp_edit.restore(model, backup)
    manifest["weights_restored_exactly"] = before == fingerprint()
    write("results/edit_manifest.json", manifest)
    write("results/experiment_config.json", {
        "model": config.MODEL, "revision": config.REVISION, "licence": "Apache-2.0",
        "precision": "float32", "device": "cpu", "seed": config.SEED,
        "random_edit_seeds": list(RANDOM_SEEDS),
        "generation": {"max_new_tokens": config.MAX_NEW_TOKENS, "do_sample": False,
                       "repetition_penalty": 1.0, "batch_size": config.BATCH_SIZE},
        "valid_reply": {"min_words": config.MIN_WORDS, "max_repetition": config.MAX_REPETITION},
        "english_system_prompt": config.ENGLISH_SYSTEM,
        "swedish_instruction": config.SWEDISH_INSTRUCTION,
        "test_prompts": {"swedish_with_english_system": N_TEST, "swedish_without_system": N_NO_SYSTEM,
                         "english": len(test["english"])},
        "probes": {k: len(v) for k, v in probes.items()},
        "torch": torch.__version__,
    })


def main():
    stages = sys.argv[1:] or ["weights", "dev", "test", "report"]
    if stages != ["report"]:
        tok, model = model_loader.load()
        if "weights" in stages:
            stage_weights(tok, model)
        if "dev" in stages:
            stage_round1(tok, model)
            stage_round2(tok, model)
            stage_round3(tok, model)
        if "test" in stages:
            stage_test(tok, model)
    if "report" in stages:
        from . import report
        report.main()


if __name__ == "__main__":
    main()
