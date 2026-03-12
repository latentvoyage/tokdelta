class BaseTokenizer:

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def token_bytes(self, token_id: int) -> bytes:
        """Return the raw byte representation of a single token."""
        raise NotImplementedError