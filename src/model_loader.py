"""Loading the model and generating replies."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def load():
    """The checkpoint on disk is never modified. Edits are applied to this in-memory copy only."""
    torch.manual_seed(config.SEED)
    tok = AutoTokenizer.from_pretrained(config.MODEL, revision=config.REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config.MODEL, revision=config.REVISION,
                                                 dtype=torch.float32)
    model.eval()
    return tok, model


def chat(prompt, system=None):
    text = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return text + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def build_items(prompts, swedish_instruction=False, settings=("no_system", "english_system")):
    """Each prompt is asked under two settings: no system prompt, and an English system prompt."""
    items = []
    for prompt in prompts:
        for setting in settings:
            system = config.ENGLISH_SYSTEM if setting == "english_system" else None
            if swedish_instruction:
                system = (system + " " if system else "") + config.SWEDISH_INSTRUCTION
            items.append({"setting": setting, "prompt": prompt, "text": chat(prompt, system)})
    return items


@torch.no_grad()
def generate(tok, model, items):
    """Greedy, batched generation. Returns copies of the items with a 'reply' field."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
    replies = [None] * len(items)
    for start in range(0, len(order), config.BATCH_SIZE):
        batch = order[start:start + config.BATCH_SIZE]
        enc = tok([items[i]["text"] for i in batch], return_tensors="pt", padding=True)
        out = model.generate(**enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id)
        for row, i in zip(out, batch):
            new_tokens = row[enc["input_ids"].shape[1]:]
            reply = tok.decode(new_tokens, skip_special_tokens=True).strip()
            # Truncated: the reply used all its tokens without the model ending its turn.
            replies[i] = (reply, tok.eos_token_id not in new_tokens.tolist())
    return [dict(item, reply=reply, truncated=truncated)
            for item, (reply, truncated) in zip(items, replies)]
