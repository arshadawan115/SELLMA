"""Pilot on the practice prompts: how often does the unchanged model fail, and how?

Run: python -m src.pilot
"""
import json
import time

from . import evaluate, model_loader


def main():
    with open("src/prompts_dev.json", encoding="utf-8") as fh:
        prompts = json.load(fh)
    tok, model = model_loader.load()
    start = time.time()
    rows = evaluate.run_generation(tok, model, prompts)
    seconds = time.time() - start
    summary = {setting: evaluate.summarize([r for r in rows if r["setting"] == setting])
               for setting in ("no_system", "english_system")}
    summary["all"] = evaluate.summarize(rows)
    summary["seconds_per_reply"] = seconds / len(rows)
    with open("results/pilot_dev.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=1)
    for name, value in summary.items():
        print(name, json.dumps(value) if isinstance(value, dict) else round(value, 1))
    for row in rows:
        if not row["clean_swedish"]:
            print("FAIL", row["setting"], row["language"], row["marker_words"], "|",
                  row["prompt"], "->", row["reply"].replace("\n", " ")[:160])


if __name__ == "__main__":
    main()
