## Fetches the appropriate tokenizer based on the tokenizer name and model name.
## Supports HuggingFace and Tiktoken; backends are imported lazily so a missing
## package only matters if you actually try to use it.


class TokenizerRegistry:

    @staticmethod
    def get_tokenizer(tokenizer_name: str, model_name: str):
        if tokenizer_name == "huggingface":
            from tokdelta.tokenizer.huggingface_tokenizer import HuggingFaceTokenizer
            return HuggingFaceTokenizer(model_name)
        elif tokenizer_name == "tiktoken":
            from tokdelta.tokenizer.tiktoken_tokenizer import TiktokenTokenizer
            return TiktokenTokenizer(model_name)
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")
    
    