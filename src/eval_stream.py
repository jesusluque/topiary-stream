"""Quality accounting for Stream artifacts: PPL per serving mode + tasks.

The discipline this repo exists to add to MoE streaming: every serving policy
ships with its measured toll. Two regimes, reported separately because they
diverge (measured: pool policies pay +6-11% PPL in teacher-forced prefill —
flat routing defeats recency — while task decode is unaffected):

  ppl     teacher-forced held-out PPL under a chosen serve mode (`exact`
          gives the model's true 4-bit base even when it cannot be loaded
          whole — the control every other number is judged against)
  tasks   HumanEval / GSM8K through the real runtime (greedy, session
          refresh between items). Report differences with McNemar in mind:
          at n=25/50, gaps of <=2 items are indistinguishable, not ties.

Usage:
    python src/eval_stream.py --artifact artifacts/qwen80-stream --stage ppl \
        --serve-mode exact --data-code data/code.jsonl --data-general data/wiki.jsonl
    python src/eval_stream.py --artifact artifacts/qwen35-stream --stage tasks \
        --humaneval 25 --gsm 50
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from math import comb
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import load_corpus, set_seeds, token_nll


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_runtime(artifact: str, pool_c: int, pool_k: int, serve_mode: str,
                 orders: str | None, floor: str | None):
    layout = json.load(open(Path(artifact) / "config.json")).get("stream_layout")
    if layout == "resident-p0":
        import fastpath as rt
        from mlx_lm import load

        model, tokenizer = load(artifact)
        mx.eval(model.parameters())
        rt.patch_fast(model, Path(artifact), pool_k)
        return model, tokenizer, rt
    import pager as rt

    model, tokenizer = rt.load_model(artifact)
    rt.patch_pool(model, Path(artifact), pool_c, pool_k, orders=orders)
    rt.S["mode"] = serve_mode
    if serve_mode == "floor2d":
        from common import find_moe_blocks

        rt.load_floor(floor, find_moe_blocks(model))
    return model, tokenizer, rt


def stage_ppl(model, tokenizer, rt, data_code: str, data_general: str,
              chunks: tuple[int, int], chunk_len: int, tag: str) -> None:
    results = {}
    for name, path, n in (("code", data_code, chunks[0]),
                          ("general", data_general, chunks[1])):
        if not path or n == 0:
            continue
        rows = load_corpus(Path(path), 10**9)[:n]
        nlls = []
        t0 = time.perf_counter()
        for row in rows:
            ids = mx.array(tokenizer.encode(row["text"])[:chunk_len])[None]
            out = model(ids)
            mx.eval(out)
            nlls.append(np.array(token_nll(out, ids)))
            del out
            rt.refresh_all()
            mx.clear_cache()
        results[f"ppl_{name}"] = float(np.exp(np.concatenate(nlls).mean()))
        print(f"[{tag}] PPL {name}: {results[f'ppl_{name}']:.4f} "
              f"({time.perf_counter() - t0:.0f}s)")
    results["peak_gb"] = mx.get_peak_memory() / 1e9
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/ppl_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[mem] peak {results['peak_gb']:.2f} GB")


def _ask(model, tokenizer, rt, prompt: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    kwargs = {"add_generation_prompt": True, "tokenize": False}
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(msgs, **kwargs)
    out = generate(model, tokenizer, text, max_tokens=max_tokens,
                   sampler=make_sampler(temp=0.0))
    rt.refresh_all()
    return out


def stage_tasks(model, tokenizer, rt, n_humaneval: int, n_gsm: int, tag: str) -> None:
    from datasets import load_dataset

    results = {}
    if n_humaneval:
        ds = load_dataset("openai/openai_humaneval", split="test")
        idx = np.random.default_rng(1234).permutation(len(ds))[:n_humaneval]
        ok = 0
        for j, i in enumerate(idx):
            r = ds[int(i)]
            prompt = ("Complete this Python function. Reply with ONLY the complete "
                      "function inside a ```python code block.\n\n```python\n"
                      + r["prompt"] + "\n```")
            out = _ask(model, tokenizer, rt, prompt, 512)
            m = re.findall(r"```(?:python)?\n(.*?)```", out, re.DOTALL)
            code = m[0] if m else out
            if r["entry_point"] not in code:
                code = r["prompt"] + code
            program = (code + "\n\n" + r["test"]
                       + f"\n\ncheck({r['entry_point']})\nprint('PASS')\n")
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(program)
            try:
                res = subprocess.run(["python3", f.name], capture_output=True,
                                     text=True, timeout=15)
                ok += int("PASS" in res.stdout)
            except Exception:
                pass
            if (j + 1) % 5 == 0:
                print(f"  humaneval {j + 1}/{n_humaneval}: {ok} ok")
        results["humaneval"] = ok / n_humaneval
        print(f"[humaneval] {ok}/{n_humaneval} = {ok / n_humaneval:.0%}")

    if n_gsm:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idx = np.random.default_rng(1234).permutation(len(ds))[:n_gsm]
        ok = 0
        for j, i in enumerate(idx):
            r = ds[int(i)]
            ref = r["answer"].split("####")[-1].strip().replace(",", "")
            out = _ask(model, tokenizer, rt,
                       r["question"] + "\n\nReason briefly and end with the final "
                       "numeric answer.", 384)
            nums = re.findall(r"-?\d[\d,]*\.?\d*", out.replace("$", ""))
            got = nums[-1].replace(",", "").rstrip(".") if nums else None
            try:
                ok += int(got is not None and abs(float(got) - float(ref)) < 1e-6)
            except ValueError:
                ok += int(got == ref)
            if (j + 1) % 10 == 0:
                print(f"  gsm {j + 1}/{n_gsm}: {ok} ok")
        results["gsm8k"] = ok / n_gsm
        print(f"[gsm8k] {ok}/{n_gsm} = {ok / n_gsm:.0%}")

    results["peak_gb"] = mx.get_peak_memory() / 1e9
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/tasks_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[mem] peak {results['peak_gb']:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality accounting for Stream artifacts")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--stage", required=True, choices=["ppl", "tasks"])
    parser.add_argument("--serve-mode", default="nosync",
                        choices=["exact", "nosync", "floor", "floor2d"])
    parser.add_argument("--pool-c", type=int, default=240)
    parser.add_argument("--pool-k", type=int, default=32)
    parser.add_argument("--orders", default=None)
    parser.add_argument("--floor", default=None)
    parser.add_argument("--data-code", default=None)
    parser.add_argument("--data-general", default=None)
    parser.add_argument("--chunks-code", type=int, default=10)
    parser.add_argument("--chunks-general", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=512)
    parser.add_argument("--humaneval", type=int, default=25)
    parser.add_argument("--gsm", type=int, default=50)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    set_seeds(1234)
    model, tokenizer, rt = load_runtime(args.artifact, args.pool_c, args.pool_k,
                                        args.serve_mode, args.orders, args.floor)
    print(f"[load] {mx.get_active_memory() / 1e9:.2f} GB")
    if args.stage == "ppl":
        stage_ppl(model, tokenizer, rt, args.data_code, args.data_general,
                  (args.chunks_code, args.chunks_general), args.chunk_len, args.tag)
    else:
        stage_tasks(model, tokenizer, rt, args.humaneval, args.gsm, args.tag)


if __name__ == "__main__":
    main()
