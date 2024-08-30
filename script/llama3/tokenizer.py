from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt


def llama_tokenizer(tokenizer_path="/home/jim/nas/yzg/Meta-Llama-3-8B-Instruct-torch/tokenizer.model"):
    # 加载分词器模型路径
    special_tokens = [
                         "<|begin_of_text|>",
                         "<|end_of_text|>",
                         "<|reserved_special_token_0|>",
                         "<|reserved_special_token_1|>",
                         "<|reserved_special_token_2|>",
                         "<|reserved_special_token_3|>",
                         "<|start_header_id|>",
                         "<|end_header_id|>",
                         "<|reserved_special_token_4|>",
                         "<|eot_id|>",  # end of turn
                     ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    tokenizer = tiktoken.Encoding(
        name=Path(tokenizer_path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
    )

    # 测试分词器编码和解码功能
    return tokenizer


if __name__ == "__main__":
    tokenizer = llama_tokenizer()
    print(tokenizer.decode(tokenizer.encode("hello world!")))
