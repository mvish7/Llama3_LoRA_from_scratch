import os
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if (128002 + i) not in self.special_tokens.values()
            }
        )

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(
        self,
        text,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=(),
        max_len: int = -1,
    ):
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        tokens += self.model.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])

        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode(self, text):
        message = {"role": "user", "content": text}

        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


if __name__ == "__main__":
    tokenizer = Tokenizer(
        "/media/vishal/workspace/projects/Llama3_LoRA_from_scratch/llama3_1_lora_fine_tune/checkpoints/Llama3.2-3B-Instruct/tokenizer.model"
    )
