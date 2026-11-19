"""Forward-pass statistics on practice prompts (no gradients)."""
import torch


@torch.no_grad()
def mean_activity(tok, model, rows):
    """Replays each row (prompt plus the reply the unchanged model gave) and averages, over the
    reply positions: the activation of every MLP neuron, and the residual stream after every layer.

    Returns (neurons: layers x neurons, residual: layers+1 x hidden, number of forward passes)."""
    layers = model.config.num_hidden_layers
    neurons = torch.zeros(layers, model.config.intermediate_size)
    residual = torch.zeros(layers + 1, model.config.hidden_size)
    state, count, hooks = {}, 0, []
    for i in range(layers):
        def hook(module, args, i=i):
            neurons[i] += args[0][0, state["start"]:].sum(0)
        hooks.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        for row in rows:
            prefix_len = tok(row["text"], return_tensors="pt")["input_ids"].shape[1]
            full = tok(row["text"] + row["reply"], return_tensors="pt")
            state["start"] = prefix_len - 1          # the position that predicts the first reply token
            out = model(**full, output_hidden_states=True)
            for i, hidden in enumerate(out.hidden_states):
                residual[i] += hidden[0, state["start"]:].sum(0)
            count += full["input_ids"].shape[1] - state["start"]
    finally:
        for h in hooks:
            h.remove()
    return neurons / count, residual / count, len(rows)


@torch.no_grad()
def neuron_stats(tok, model, rows, layer):
    """Mean and standard deviation of every MLP neuron in one layer, over all positions (prompt and
    reply, start-of-text token excluded) of the replayed practice rows."""
    collected = []
    hook = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
        lambda module, args: collected.append(args[0][0, 1:].clone()))
    try:
        for row in rows:
            model(**tok(row["text"] + row["reply"], return_tensors="pt"))
    finally:
        hook.remove()
    activations = torch.cat(collected)
    return activations.mean(0), activations.std(0)
