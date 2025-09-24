import torch


def sample(logits, temperature: float = 1.0, top_k: int | None = None):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.
        top_k (int, optional): Top_k for cropping logits. Defaults to None.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    probs = torch.softmax(logits, dim=-1)
    # return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()
