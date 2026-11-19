"""Gradient-free weight edits on the MLP output matrices (down_proj), plus the random control.

One edit = for a set of (layer, column) pairs, remove a direction d from that column:
    W[:, c] <- W[:, c] - beta * (d . W[:, c]) * d
Column c of down_proj is what MLP neuron c writes into the residual stream, so the edit stops
(beta = 1) or reverses (beta = 2) that neuron's push along d.
"""
import torch

from . import config


def token_ids(tok, words):
    ids = []
    for word in words:
        for form in {word, word.capitalize()}:
            pieces = tok.encode("x " + form, add_special_tokens=False)[1:]
            if len(pieces) == 1:
                ids.append(pieces[0])
    return sorted(set(ids))


def non_swedish_direction(tok, model, pairs=None):
    """Unit direction in the residual stream that raises the output scores of Norwegian, Danish
    and English function words relative to their Swedish counterparts.

    Uses only the output embedding matrix and the final norm gain: no forward pass."""
    pairs = config.DIRECTION_PAIRS + config.ENGLISH_PAIRS if pairs is None else pairs
    unembed = model.lm_head.weight.detach()
    swedish = token_ids(tok, [sv for sv, _ in pairs])
    other = token_ids(tok, [nd for _, nd in pairs])
    direction = (unembed[other].mean(0) - unembed[swedish].mean(0)) * model.model.norm.weight.detach()
    return direction / direction.norm()


def top_tokens(tok, model, vector, k=20):
    """Tokens whose output score rises most when `vector` is added to the final residual."""
    scores = model.lm_head.weight.detach() @ vector
    return [tok.decode([i]) for i in torch.topk(scores, k).indices.tolist()]


def down_proj(model, layer):
    return model.model.layers[layer].mlp.down_proj.weight   # shape (hidden, neurons)


def column_projections(model, direction):
    """d . W[:, c] for every MLP neuron: tensor of shape (layers, neurons).
    `direction` is one vector for all layers, or a (layers, hidden) matrix with one per layer."""
    return torch.stack([(direction[i] if direction.dim() == 2 else direction) @ down_proj(model, i).detach()
                        for i in range(model.config.num_hidden_layers)])


def top_columns(scores, k):
    flat = torch.topk(scores.flatten(), k).indices
    return [(int(i // scores.shape[1]), int(i % scores.shape[1])) for i in flat]


def apply_edit(model, columns, direction, beta=1.0, target_norm=None):
    """Applies the edit in memory. Returns (backup, norm of the change).
    With target_norm the change is rescaled to that size (used by the random control)."""
    deltas = []
    for layer, column in columns:
        weight = down_proj(model, layer).data
        d = direction[layer] if direction.dim() == 2 else direction   # one direction per layer, or one for all
        deltas.append(-beta * (d @ weight[:, column]) * d)
    norm = torch.stack(deltas).norm().item()
    scale = 1.0 if target_norm is None else target_norm / max(norm, 1e-12)
    backup = []
    for (layer, column), delta in zip(columns, deltas):
        weight = down_proj(model, layer).data
        backup.append((layer, column, weight[:, column].clone()))
        weight[:, column] += scale * delta
    return backup, norm * scale


def restore(model, backup):
    for layer, column, original in backup:
        down_proj(model, layer).data[:, column] = original


def random_edit(model, like_columns, seed, target_norm):
    """Control: random columns (same number per layer) and a random direction, rescaled to the
    same size as the targeted edit."""
    generator = torch.Generator().manual_seed(seed)
    neurons = model.config.intermediate_size
    columns = []
    for layer in sorted({l for l, _ in like_columns}):
        count = sum(1 for l, _ in like_columns if l == layer)
        columns += [(layer, c) for c in torch.randperm(neurons, generator=generator)[:count].tolist()]
    direction = torch.randn(model.config.hidden_size, generator=generator)
    direction = direction / direction.norm()
    backup, norm = apply_edit(model, columns, direction, 1.0, target_norm)
    return backup, norm, columns


def apply_constant_push(model, layer, neurons, means, vector):
    """The targeted edit of this study. It writes a constant push into the residual stream by
    using MLP neurons that fire at an almost constant level m_c:

        W[:, c] <- W[:, c] + vector / (n * m_c)      for each of the n chosen neurons c

    When the neurons fire at their usual level they add `vector` in total. Returns
    (backup, norm of the change)."""
    weight = down_proj(model, layer).data
    backup, total = [], 0.0
    for c in neurons:
        delta = vector / (len(neurons) * float(means[c]))
        backup.append((layer, c, weight[:, c].clone()))
        weight[:, c] += delta
        total += float(delta.norm() ** 2)
    return backup, total ** 0.5


def manifest(model, name, method, columns, beta, norm, extra=None):
    per_layer = {}
    for layer, _ in columns:
        per_layer[str(layer)] = per_layer.get(str(layer), 0) + 1
    weights = len(columns) * model.config.hidden_size
    entry = {
        "name": name, "method": method,
        "tensor": "model.layers.{layer}.mlp.down_proj.weight[:, column]",
        "columns": [list(c) for c in columns], "columns_per_layer": per_layer,
        "beta": beta, "weights_changed": weights,
        "share_of_model_pct": 100 * weights / sum(p.numel() for p in model.parameters()),
        "change_norm": norm,
    }
    entry.update(extra or {})
    return entry
