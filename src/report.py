"""Turns the saved test results into tables, example replies and one chart."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import evaluate  # noqa: E402

LABELS = {
    "baseline": "Unchanged model",
    "swedish_prompt": "Told to answer in Swedish",
    "edit_targeted": "Targeted weight edit",
    "random_edit_1": "Random edit, seed 1",
    "random_edit_2": "Random edit, seed 2",
    "random_edit_3": "Random edit, seed 3",
    "random_edit_4": "Random edit, seed 4 (added)",
    "random_edit_5": "Random edit, seed 5 (added)",
    "random_edit_6": "Random edit, seed 6 (added)",
    "steering": "Steering while running (extra)",
    "edit_weights_only": "Weights-only edit (extra)",
}


def load(name):
    path = f"results/test/{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(values):
    return 100 * sum(values) / len(values)


def main():
    results = {key: load(key) for key in LABELS}
    results = {key: value for key, value in results.items() if value}
    baseline = results["baseline"]
    base_valid = [float(r["valid_swedish"]) for r in baseline["rows_english_system"]]
    base_knowledge = [float(x) for x in baseline["knowledge_correct"]]
    rows = []
    for key, result in results.items():
        valid = [float(r["valid_swedish"]) for r in result["rows_english_system"]]
        knowledge = [float(x) for x in result["knowledge_correct"]]
        change = [v - b for v, b in zip(valid, base_valid)]
        knowledge_change = [k - b for k, b in zip(knowledge, base_knowledge)]
        rows.append({
            "condition": key, "label": LABELS[key],
            "valid_swedish_pct": pct(valid), "valid_ci": evaluate.bootstrap_ci(valid),
            "change_points": pct(change), "change_ci": evaluate.bootstrap_ci(change),
            "swedish_pct": result["english_system"]["swedish_pct"],
            "mean_words": result["english_system"]["mean_words"],
            "truncated_pct": result["english_system"]["truncated_pct"],
            "repetition_pct": result["english_system"]["repetition_pct"],
            "knowledge_pct": pct(knowledge), "knowledge_ci": evaluate.bootstrap_ci(knowledge),
            "knowledge_change_points": pct(knowledge_change),
            "knowledge_change_ci": evaluate.bootstrap_ci(knowledge_change),
            "general_pct": result["general_pct"],
            "grammar_pct": result["grammar_pct"],
            "no_system_swedish_pct": result["no_system"]["valid_swedish_pct"],
            "english_users_english_pct": result["english_users_answered_in_english_pct"],
        })
    with open("results/metrics.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    header = ["Condition", "Valid Swedish replies % (95% interval)", "Change vs unchanged, points (95% interval)",
              "Any Swedish %", "Mean words", "Truncated %", "Swedish knowledge %", "General questions %",
              "Grammar pairs %", "Valid Swedish %, no system prompt", "English users answered in English %"]
    table = []
    for r in rows:
        change = "" if r["condition"] == "baseline" else (
            f"{r['change_points']:+.1f} ({r['change_ci'][0]:+.1f} to {r['change_ci'][1]:+.1f})")
        table.append([r["label"],
                      f"{r['valid_swedish_pct']:.1f} ({r['valid_ci'][0]:.1f} to {r['valid_ci'][1]:.1f})",
                      change, f"{r['swedish_pct']:.1f}", f"{r['mean_words']:.1f}", f"{r['truncated_pct']:.0f}",
                      f"{r['knowledge_pct']:.1f}", f"{r['general_pct']:.1f}", f"{r['grammar_pct']:.1f}",
                      f"{r['no_system_swedish_pct']:.1f}", f"{r['english_users_english_pct']:.1f}"])
    with open("results/metrics_summary.csv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([header] + table)
    markdown = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    markdown += ["| " + " | ".join(row) + " |" for row in table]
    with open("results/metrics_table.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(markdown) + "\n")
    print("\n".join(markdown))

    # Example replies. First every test prompt where the unchanged model failed, then every prompt
    # where a condition turned a valid Swedish reply into something else.
    lines = ["# Example replies (held-out test prompts, English system prompt)", "",
             "Replies are cut at 40 tokens. In brackets: detected language, and whether the reply "
             "counts as a valid Swedish reply.", ""]
    n = len(baseline["rows_english_system"])
    failed = [i for i in range(n) if not baseline["rows_english_system"][i]["valid_swedish"]]
    broken = [i for i in range(n) if baseline["rows_english_system"][i]["valid_swedish"]
              and any(not r["rows_english_system"][i]["valid_swedish"] for r in results.values())]
    for title, indices in (("Prompts where the unchanged model did not give a valid Swedish reply", failed),
                           ("Prompts where a condition broke a valid Swedish reply", broken)):
        lines += [f"## {title} ({len(indices)})", ""]
        for i in indices:
            lines += [f"**Prompt:** {baseline['rows_english_system'][i]['prompt']}", ""]
            for key, result in results.items():
                row = result["rows_english_system"][i]
                mark = "valid" if row["valid_swedish"] else "not valid"
                lines.append(f"- {LABELS[key]} [{row['language']}, {mark}]: " + row["reply"].replace("\n", " "))
            lines.append("")
    with open("results/sample_outputs.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    ink, muted, blue, gray, surface = "#0b0b0b", "#52514e", "#2a78d6", "#a9a8a2", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, facecolor=surface)
    order = rows[::-1]
    panels = [("valid_swedish_pct", "valid_ci", "Valid Swedish replies (%)", "Swedish question, English system prompt"),
              ("knowledge_pct", "knowledge_ci", "Questions answered correctly (%)", "Swedish knowledge questions")]
    for ax, (value, interval, xlabel, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        for i, r in enumerate(order):
            ax.barh(i, r[value], height=0.56, color=gray if r["condition"] == "baseline" else blue, zorder=2)
            ax.plot(r[interval], [i, i], color=ink, linewidth=1.2, zorder=3)
            ax.text(max(r[interval][1], r[value]) + 1.5, i, f"{r[value]:.0f}", va="center", fontsize=9, color=ink)
        ax.axvline(order[-1][value], color=muted, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(0, 112)
        ax.set_xlabel(xlabel, color=muted, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([r["label"] for r in order], color=ink, fontsize=9)
    fig.text(0.01, 0.01, "Held-out test set. Black lines: 95% bootstrap interval. Dashed line: unchanged model.",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("results/main_results.png", dpi=200, facecolor=surface)


if __name__ == "__main__":
    main()
