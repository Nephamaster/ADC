import argparse
import json
import os

from typing import Any

from transformers import AutoTokenizer


def bytes_to_unicode() -> dict[int, str]:
    """
    Byte-level unicode map used by GPT-style BPE tokenizers.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 172 + 1))
        + list(range(174, 255 + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_byte_decoder(tokenizer) -> dict[str, int]:
    if hasattr(tokenizer, "byte_decoder"):
        return tokenizer.byte_decoder
    return {v: k for k, v in bytes_to_unicode().items()}


def decode_bpe_piece(piece: str, byte_decoder: dict[str, int]) -> str:
    try:
        return bytearray([byte_decoder[c] for c in piece]).decode("utf-8", errors="replace")
    except Exception:
        # Fallback for non-bytelevel or already-decoded tokens.
        return piece


def is_chinese_char(c: str) -> bool:
    if len(c) != 1:
        return False
    cp = ord(c)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0x2CEB0 <= cp <= 0x2EBEF
        or 0x30000 <= cp <= 0x3134F
    )


def is_chinese_string(s: str) -> bool:
    return bool(s) and all(is_chinese_char(c) for c in s)


def normalize_vocab(vocab_like: Any) -> dict[str, int]:
    if isinstance(vocab_like, dict):
        return {str(k): int(v) for k, v in vocab_like.items()}
    if isinstance(vocab_like, list):
        normalized: dict[str, int] = {}
        for i, item in enumerate(vocab_like):
            if isinstance(item, list):
                token = item[0]
            else:
                token = item
            normalized[str(token)] = i
        return normalized
    raise ValueError("Unsupported vocab format")


def normalize_merges(merges_like: Any) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for item in merges_like or []:
        if isinstance(item, str):
            a, b = item.split(" ", 1)
            merges.append((a, b))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            merges.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"Unsupported merge item: {item}")
    return merges


def extract_bpe_state(tokenizer):
    """
    Return (tokenizer_json, vocab, merges), where tokenizer_json can be None on legacy tokenizers.
    """
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
        model = tokenizer_json.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(f"Only BPE tokenizers are supported, got: {model.get('type')}")
        vocab = normalize_vocab(model.get("vocab", {}))
        merges = normalize_merges(model.get("merges", []))
        return tokenizer_json, vocab, merges

    if hasattr(tokenizer, "encoder") and hasattr(tokenizer, "bpe_ranks"):
        vocab = {str(k): int(v) for k, v in tokenizer.encoder.items()}
        merges_with_rank = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])
        merges = [(str(a), str(b)) for (a, b), _rank in merges_with_rank]
        return None, vocab, merges

    raise ValueError("Cannot extract BPE state from tokenizer")


def build_reindexed_vocab_and_mapping(vocab: dict[str, int], removed_token_ids: set[int]) -> tuple[dict[str, int], dict[int, int]]:
    """
    Reindex retained tokens to contiguous ids [0, new_vocab_size), preserving old id order.
    Returns:
      - new_vocab: token -> new_id
      - new2old: new_id -> old_id
    """
    kept = sorted(
        ((token, old_id) for token, old_id in vocab.items() if old_id not in removed_token_ids),
        key=lambda x: x[1],
    )
    new_vocab: dict[str, int] = {}
    new2old: dict[int, int] = {}
    for new_id, (token, old_id) in enumerate(kept):
        new_vocab[token] = new_id
        new2old[new_id] = old_id
    return new_vocab, new2old


def prune(old_model_path: str, new_model_path: str, prune_vocab_tokens: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(old_model_path)
    tokenizer_json, vocab, merges = extract_bpe_state(tokenizer)
    byte_decoder = get_byte_decoder(tokenizer)

    need_to_delete_words = set()
    need_to_delete_word_ids = set()
    for token, token_id in vocab.items():
        decoded = decode_bpe_piece(token, byte_decoder)
        if len(decoded) > 1 and is_chinese_string(decoded):
            need_to_delete_words.add(decoded)
            need_to_delete_word_ids.add(token_id)
            print(f"DELETE WORD {decoded}")
    print(len(need_to_delete_word_ids), "chinese words to delete")

    removed_token_strs = {token for token, token_id in vocab.items() if token_id in need_to_delete_word_ids}
    need_to_delete_merge_idx = set()
    for merge_idx, (a, b) in enumerate(merges):
        merged = a + b
        da = decode_bpe_piece(a, byte_decoder)
        db = decode_bpe_piece(b, byte_decoder)
        dab = decode_bpe_piece(merged, byte_decoder)
        if (
            dab in need_to_delete_words
            or da in need_to_delete_words
            or db in need_to_delete_words
            or a in removed_token_strs
            or b in removed_token_strs
            or merged in removed_token_strs
        ):
            need_to_delete_merge_idx.add(merge_idx)
            print(f"DELETE MERGE {dab}")
    print(len(need_to_delete_merge_idx), "merges to delete")

    if prune_vocab_tokens:
        pruned_vocab, new2old = build_reindexed_vocab_and_mapping(vocab, need_to_delete_word_ids)
    else:
        pruned_vocab = vocab
        new2old = {old_id: old_id for old_id in sorted(vocab.values())}
    pruned_merges = [merge for i, merge in enumerate(merges) if i not in need_to_delete_merge_idx]

    os.makedirs(new_model_path, exist_ok=True)
    tokenizer.save_pretrained(new_model_path)

    with open(os.path.join(new_model_path, "vocab.json"), "wt", encoding="utf-8") as f:
        json.dump(pruned_vocab, f, ensure_ascii=False)
    with open(os.path.join(new_model_path, "merges.txt"), "wt", encoding="utf-8") as f:
        f.write("\n".join(f"{a} {b}" for a, b in pruned_merges))
        f.write("\n")
    with open(os.path.join(new_model_path, "new2old_token_id.json"), "wt", encoding="utf-8") as f:
        json.dump(new2old, f, ensure_ascii=False, indent=2)

    # New Transformers loads tokenizer.json first when present, so patch it too.
    if tokenizer_json is not None:
        tokenizer_json["model"]["vocab"] = pruned_vocab
        tokenizer_json["model"]["merges"] = [f"{a} {b}" for a, b in pruned_merges]
        with open(os.path.join(new_model_path, "tokenizer.json"), "wt", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False)


def main(old_model_path: str, new_model_path: str):
    text = "\u8fd9\u662f\u4e00\u4e2a\u4e2d\u6587\u5206\u8bcd\u6d4b\u8bd5"
    tokenizer_old = AutoTokenizer.from_pretrained(old_model_path)
    print([tokenizer_old.decode(_id) for _id in tokenizer_old.encode(text)])
    print(len(tokenizer_old))

    tokenizer_new = AutoTokenizer.from_pretrained(new_model_path)
    print([tokenizer_new.decode(_id) for _id in tokenizer_new.encode(text)])
    print(len(tokenizer_new))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--new_model_path", type=str, required=True)

    args = parser.parse_args()
    prune(args.old_model_path, args.new_model_path, prune_vocab_tokens=True)
    main(args.old_model_path, args.new_model_path)
