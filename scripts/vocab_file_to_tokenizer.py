"""Script for creating a simple whitespace tokenizer over a predefined vocabulary and
saving it to a JSON file using HuggingFace's tokenizers library.

The resulting tokenizer simply and naively splits on whitespace. It will add unknown and
padding tokens as appropriate, but will not add any other special tokens, e.g.
beginning/end-of-sentence tokens. However, you may (and should) specify any special
tokens that may already appear in the corpora that this tokenizer will tokenize. Thus,
the resultant tokenizer is intended for use with corpora that are already tokenized.
"""

import argparse
from argparse import ArgumentParser
from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

PARSER_CONFIG = {
    "description": __doc__,
}


def construct_cli_parser() -> ArgumentParser:
    """Build the CLI argument parser for this function.

    Returns:
        An `ArgumentParser` instance with the appropriate settings.
    """
    parser = ArgumentParser(**PARSER_CONFIG)

    parser.add_argument("f_in", help="Path to vocab file.")
    parser.add_argument("f_out", help="Path to JSON file where to save the tokenizer.")
    parser.add_argument("-p", "--pad_token", required=True, help="Padding token")
    parser.add_argument("-u", "--unk_token", required=True, help="Unknown token")
    parser.add_argument(
        "-s",
        "--other_special_tokens",
        nargs="*",
        default=[],
        help="Any additional special tokens.",
    )
    parser.add_argument(
        "-l",
        "--pad_left",
        action="store_true",
        help="Pad on the left (rather than the default, the right).",
    )

    return parser


def construct_vocab(f_in: str, special_tokens: list[str] = []) -> dict[str, int]:
    """Read in a plaintext vocabulary file and convert it into a dictionary format.

    Args:
        f_in: The vocab file. Should consist of a list of tokens, with one token per
            line.
        special_tokens: A list of special tokens, e.g. padding, unknown, bos/eos, etc.
            These will be assigned the first integer id's, before all non-special
            tokens, in the order that they are given in this list.
    Returns:
        A dictionary whose keys are tokens in the vocabulary and whose values are the
        corresponding unique integer id's associated with each word. The id's run from 0
        to N-1 for a vocabulary of size N (including all special tokens).
    """
    with open(f_in, "r") as f:
        vocab_file_words = (
            word for word in (word.strip() for word in f.readlines()) if word
        )

        # We could use a set below, but then we couldn't put special tokens first
        # maybe we wanna make this a setting via CLI argument eventually?
        all_words = special_tokens + [
            word for word in vocab_file_words if word not in special_tokens
        ]

    return {word: idx for idx, word in enumerate(all_words)}

def construct_tokenizer(
    f_in: str,
    pad_token: str,
    unk_token: str,
    other_special_tokens: list[str] = [],
    pad_left: bool = False,
) -> Tokenizer:
    """Build the output `Tokenizer` object.

    Args:
        f_in: Path to the input vocabulary file, containing the tokens in the
            vocabulary, with one token per line.
        pad_token: The padding token to use, as a string.
        unk_token: The unknown token to use, as a string.
        other_special_tokens: A list of any other special tokens in the vocabulary, as
            strings.
        pad_left: If `True`, tokenizer will pad from the left; otherwise, it will pad
            from the right.
            
            Default: `False`.
    Returns:
        The `tokenizers.Tokenizer` instance constructed from the vocabulary in the given
        file with the given special tokens/padding settings.
    """
    special_tokens = [pad_token, unk_token] + other_special_tokens

    vocab = construct_vocab(f_in, special_tokens)
    tok = Tokenizer(model=WordLevel(vocab=vocab, unk_token=unk_token))
    tok.pre_tokenizer = WhitespaceSplit()

    tok.enable_padding(direction=["right", "left"][pad_left], pad_token=pad_token)
    tok.add_special_tokens(special_tokens)

    return tok


def main() -> None:
    parser = construct_cli_parser()
    args = parser.parse_args()

    tok_args = vars(args)
    f_out = tok_args.pop("f_out")

    print("Building tokenizer...")
    tok = construct_tokenizer(**tok_args)
    print(f"Built tokenizer. Saving to {f_out}.")
    tok.save(f_out)


if __name__ == "__main__":
    main()
