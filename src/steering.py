"""Activation steering: add a fixed vector to the residual stream while the model runs.
The weights are not changed."""


class Steering:
    def __init__(self, model, layer, vector):
        self.module = model.model.layers[layer]
        self.vector = vector
        self.handle = None

    def __enter__(self):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (output[0] + self.vector,) + tuple(output[1:])
            return output + self.vector
        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
