import torch
import torch.nn.functional as F


def compute_loss(model, inp1, inp2, **flags):
    """
    Compute log loss (float32 scalar) as in the paper
    """

    mask = model.inp_voc.compute_mask(inp1)  # [batch_size, out_len]
    targets_1hot = F.one_hot(inp1, len(model.inp_voc)).to(torch.float32)

    # outputs of the model, [batch_size, out_len, num_tokens]
    logits_seq = model(inp1, inp2)

    # log-probabilities of all tokens at all steps, [batch_size, out_len, num_tokens]
    logprobs_seq = -torch.log_softmax(logits_seq, dim=-1)

    # log-probabilities of correct outputs, [batch_size, out_len]
    logp_out = (logprobs_seq * targets_1hot).sum(dim=-1)

    return logp_out[mask].sum(dim=-1) / mask.sum()  # average loss, scalar
