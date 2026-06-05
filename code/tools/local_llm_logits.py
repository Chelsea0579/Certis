"""Score-token logit capture helper.

Given a prompt that ends with the question, append " Score: " and inspect the
LLM's softmax probability for token "1" vs token "0" at that next position.
Returns a continuous confidence value c  in  [0, 1] = P(token=1) / (P(1) + P(0)).
"""
import os, torch
import sys
sys.path.insert(0, os.path.dirname(__file__))
import local_llm_patch as P


_TOKEN_IDS = {}  # tokenizer-id -> (id_of_1, id_of_0)


def _get_token_ids():
    tok, _ = P._load_local_llm()
    key = id(tok)
    if key in _TOKEN_IDS:
        return _TOKEN_IDS[key]
    # Try a few common tokenizations of "1" and "0"
    candidates_1 = ["1", " 1"]
    candidates_0 = ["0", " 0"]
    id1 = None; id0 = None
    for s in candidates_1:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            id1 = ids[0]; break
    for s in candidates_0:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            id0 = ids[0]; break
    if id1 is None or id0 is None:
        # Fallback: get any tokenization for "1" / "0" - pick last token
        id1 = tok.encode("1", add_special_tokens=False)[-1]
        id0 = tok.encode("0", add_special_tokens=False)[-1]
    _TOKEN_IDS[key] = (id1, id0)
    return id1, id0


def score_confidence(prompt: str, score_prefix: str = "\nScore:") -> float:
    """Returns P(next_token=='1') / (P('1') + P('0')) - clamped to [0, 1]."""
    tok, mdl = P._load_local_llm()
    id1, id0 = _get_token_ids()
    msgs = [{"role": "user", "content": prompt}]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if os.environ.get("CERTIS_DISABLE_THINKING") == "1":
        chat_kwargs["enable_thinking"] = False
    try:
        text = tok.apply_chat_template(msgs, **chat_kwargs)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = text + score_prefix  # force the model to predict the score token next
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(mdl.device)
    with torch.inference_mode():
        out = mdl(**inputs)
    logits = out.logits[0, -1]  # logits for the next token
    # P(1) and P(0) via softmax (normalize over just these two for stability)
    log1 = logits[id1].item(); log0 = logits[id0].item()
    import math
    lmax = max(log1, log0)
    p1 = math.exp(log1 - lmax); p0 = math.exp(log0 - lmax)
    return p1 / (p1 + p0)
