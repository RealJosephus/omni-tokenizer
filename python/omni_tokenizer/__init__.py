"""VQ tokenization through the native omni-tokenizer library."""

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"Tokenizer", "TokenizationError"}:
        from .runtime import TokenizationError, Tokenizer

        return {"Tokenizer": Tokenizer, "TokenizationError": TokenizationError}[name]
    raise AttributeError(name)


__all__ = ["Tokenizer", "TokenizationError", "__version__"]
