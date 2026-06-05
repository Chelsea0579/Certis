"""Drop-in replacement for langchain_community.llms.Ollama so the verification
pipeline runs with a local HF model on GPU. Activates by setting CERTIS_LOCAL_LLM=<path>.

This module patches experiment.prep_llm.prompt_answer to call the local model
directly, bypassing the Ollama+LangChain pipe (which expects an Ollama daemon).
"""
import os, sys, threading
import torch

_LLM_CACHE = {}
_LOCK = threading.Lock()

def _load_local_llm():
    path = os.environ["CERTIS_LOCAL_LLM"]
    with _LOCK:
        if path in _LLM_CACHE:
            return _LLM_CACHE[path]
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[local_llm_patch] loading {path}", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        mdl.eval()
        _LLM_CACHE[path] = (tok, mdl)
        print(f"[local_llm_patch] loaded; device map: {getattr(mdl, 'hf_device_map', {})}", flush=True)
        return tok, mdl

def _generate(prompt: str, max_new_tokens: int = 64) -> str:
    tok, mdl = _load_local_llm()
    msgs = [{"role": "user", "content": prompt}]
    # Allow disabling thinking mode for Qwen3 (set CERTIS_DISABLE_THINKING=1)
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if os.environ.get("CERTIS_DISABLE_THINKING") == "1":
        chat_kwargs["enable_thinking"] = False
    try:
        text = tok.apply_chat_template(msgs, **chat_kwargs)
    except TypeError:
        # older tokenizers don't accept enable_thinking
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    # Bump max_new_tokens for thinking-mode models if env is large
    eff_max = int(os.environ.get("CERTIS_MAX_NEW_TOKENS", max_new_tokens))
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(mdl.device)
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            max_new_tokens=eff_max,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)

def patch_prep_llm():
    """Monkey-patch experiment.prep_llm.prompt_answer."""
    from langchain_core.prompts import ChatPromptTemplate
    from experiment import prep_llm as pl

    def prompt_answer(prompt_template: str, **kwargs) -> str:
        # mimic the original LangChain ChatPromptTemplate behavior
        prompt = ChatPromptTemplate.from_template(prompt_template)
        text = prompt.format(**kwargs)
        return _generate(text)

    pl.prompt_answer = prompt_answer
    print("[local_llm_patch] prompt_answer patched to local LLM", flush=True)
