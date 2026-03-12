import tiktoken
from tokdelta.tokenizer.base_tokenizer import BaseTokenizer


class TiktokenTokenizer(BaseTokenizer):

    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        return self.tokenizer.decode_single_token_bytes(token_id)