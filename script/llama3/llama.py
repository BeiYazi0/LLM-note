import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from tqdm import tqdm

from datasets import load_dataset

from rope import ROPEAttention
from tokenizer import llama_tokenizer


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 1024, ffn_dim_multiplier: float = 1.3):
        super().__init__()
        hidden_size = int(ffn_dim_multiplier * 8 * dim // 3)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, dim)
        self.w3 = nn.Linear(dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLamaDecoder(nn.Module):
    def __init__(self, hidden_size, multiple_of=1024, ffn_dim_multiplier=1.3, n_heads=32):
        super().__init__()

        self.rms1 = RMSNorm(hidden_size)
        self.multi_head = ROPEAttention(hidden_size=hidden_size, n_heads=n_heads)
        self.rms2 = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, multiple_of, ffn_dim_multiplier)

    def forward(self, x):
        batch_size, num_steps, _ = x.shape
        mask = torch.full((num_steps, num_steps), 1, device=x.device)
        mask = 1 - torch.triu(mask, diagonal=1)

        norm_x = self.rms1(x)
        attn_output = self.multi_head(norm_x, attn_mask=mask)

        unnorm_attn = x + attn_output
        norm_attn = self.rms2(unnorm_attn)
        return unnorm_attn + self.ffn(norm_attn)


class LLama(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_size, hidden_size, n_layers = config["vocab_size"], config["dim"], config["n_layers"]
        multiple_of, ffn_dim_multiplier, n_heads = config["multiple_of"], config["ffn_dim_multiplier"], config[
            "n_heads"]

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.blks = nn.Sequential()
        for i in range(n_layers):
            self.blks.add_module("block" + str(i), LLamaDecoder(hidden_size, multiple_of, ffn_dim_multiplier, n_heads))

        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        rep = self.blks(emb)

        return self.dense(rep)


class LLamaModel:
    def __init__(self, path="/home/jim/nas/yzg/Meta-Llama-3-8B-Instruct-torch/consolidated.00.pth"):
        device = torch.device("cuda:0")
        self.tokenizer = llama_tokenizer("/home/jim/nas/yzg/Meta-Llama-3-8B-Instruct-torch/tokenizer.model")

        with open("/home/jim/nas/yzg/Meta-Llama-3-8B-Instruct-torch/params.json", "r") as f:
            config = json.load(f)

        print(config)
        self.model = LLama(config)  # .to(device)
        self.load_weight(self.model, path)

        # print(self.model)
        self.infer("the answer to the ultimate question of life, the universe, and everything is ")

    def infer(self, inputs):
        tokens = [128000] + self.tokenizer.encode(inputs)
        print(tokens)

        tokens = torch.unsqueeze(torch.tensor(tokens), dim=0)

        out = self.model(tokens)
        next_token = torch.argmax(torch.squeeze(out, dim=0)[-1])
        print(next_token)

        res = self.tokenizer.decode([next_token.item()])
        print(res)

    @staticmethod
    @torch.no_grad()
    def load_weight(model, path):
        weights = torch.load(path)
        # weights = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
        model.embedding.weight.data.copy_(weights["tok_embeddings.weight"])

        with tqdm(total=32, desc='Load', postfix=dict, mininterval=0.3) as pbar:
            for i, decoder in enumerate(model.blks.children()):
                decoder.multi_head.wq.weight.data.copy_(weights[f"layers.{i}.attention.wq.weight"])
                decoder.multi_head.wk.weight.data.copy_(weights[f"layers.{i}.attention.wk.weight"])
                decoder.multi_head.wv.weight.data.copy_(weights[f"layers.{i}.attention.wv.weight"])
                decoder.multi_head.wo.weight.data.copy_(weights[f"layers.{i}.attention.wo.weight"])

                decoder.rms1.weight.data.copy_(weights[f"layers.{i}.attention_norm.weight"])
                decoder.rms2.weight.data.copy_(weights[f"layers.{i}.ffn_norm.weight"])

                decoder.ffn.w1.weight.data.copy_(weights[f"layers.{i}.feed_forward.w1.weight"])
                decoder.ffn.w2.weight.data.copy_(weights[f"layers.{i}.feed_forward.w2.weight"])
                decoder.ffn.w3.weight.data.copy_(weights[f"layers.{i}.feed_forward.w3.weight"])

                pbar.update(1)

        model.dense.weight.data.copy_(weights["output.weight"])
        del weights

    def calculate_perplexity(self, texts):
        total_loss = 0
        total_length = 0
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        with tqdm(total=len(texts), desc='test ppl', postfix=dict, mininterval=0.3) as pbar:
            for text in texts:
                inputs = [128000] + self.tokenizer.encode(text)
                inputs = torch.unsqueeze(torch.tensor(inputs), dim=0)
                with torch.no_grad():
                    outputs = self.model(inputs)

                lm_logits = torch.softmax(outputs, dim=-1)

                shift_logits = lm_logits[:, 1:-1, :].contiguous()
                shift_labels = inputs[:, 2:].contiguous()

                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                total_length += inputs.size(1)

                pbar.set_postfix(**{'total_loss': total_loss,
                                    'total_length': total_length,
                                    'avg_loss': total_loss / total_length})
                pbar.update(1)

        perplexity = torch.exp(torch.tensor(total_loss / total_length))
        return perplexity.item()


if __name__ == "__main__":
    1# 准备PTB数据集
    dataset = load_dataset("ptb_text_only", "penn_treebank")
    test_data = dataset["test"]["sentence"]
    print(test_data[0])

    model = LLamaModel()

    perplexity = model.calculate_perplexity(test_data)
    print(f"Perplexity on PTB test set: {perplexity}")
