import torch.nn.functional as F
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    print(matmul_qk.shape)
    dk = q.size()[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask  # (seq_len, seq_len)


# Example usage
d_model = 7
batch_size = 2
seq_len = 3

q = torch.rand((batch_size, seq_len, d_model))
k = torch.rand((batch_size, seq_len, d_model))
v = torch.rand((batch_size, seq_len, d_model))
mask = create_look_ahead_mask(seq_len)

attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
print(k.transpose(-2, -1).shape)
print(attention_output)