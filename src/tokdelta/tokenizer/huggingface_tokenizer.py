from transformers import AutoTokenizer
from tokdelta.tokenizer.base_tokenizer import BaseTokenizer


class HuggingFaceTokenizer(BaseTokenizer):

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        piece = self.tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            return b""
        # HF tokenizers sometimes use the unicode replacement char for bytes;
        # convert_tokens_to_string handles that for us.
        return self.tokenizer.convert_tokens_to_string([piece]).encode("utf-8")