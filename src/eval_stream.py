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
                 orders: str | None, floor: str | None,
                 centroid: str = "uniform", p1_frac: str | float = 1.0):
    layout = json.load(open(Path(artifact) / "config.json")).get("stream_layout")
    if layout is None:
        # checkpoint normal (p.ej. el campeón Topiary estático): sin runtime
        try:  # anchos per-layer de topiary (shim del lab, vía PYTHONPATH)
            from dense_loader import maybe_patch_per_layer
            maybe_patch_per_layer(artifact)
        except ImportError:
            pass
        from mlx_lm import load

        class _PlainRT:
            @staticmethod
            def refresh_all():
                return 0

        model, tokenizer = load(artifact)
        mx.eval(model.parameters())
        return model, tokenizer, _PlainRT
    if layout == "resident-p0":
        import fastpath as rt
        from mlx_lm import load

        try:  # anchos per-layer también en artefactos stream (30B taper)
            from dense_loader import maybe_patch_per_layer
            maybe_patch_per_layer(artifact)
        except ImportError:
            pass
        model, tokenizer = load(artifact)
        mx.eval(model.parameters())
        rt.patch_fast(model, Path(artifact), pool_k, centroid, p1_frac)
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


OPENAI_BASE: str | None = None   # --openai-base: baseline externo (llama-server)


def _ask(model, tokenizer, rt, prompt: str, max_tokens: int) -> str:
    if OPENAI_BASE:
        # Baseline externo vía API OpenAI-compatible (p.ej. llama-server con
        # el UD-Q2 de Unsloth): mismos prompts, greedy, mismo parser.
        import urllib.request

        body = json.dumps({"model": "baseline", "temperature": 0.0,
                           "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(OPENAI_BASE.rstrip("/") + "/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]
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


def stage_bench(model, tokenizer, rt, benches: list[str], n_bench: int,
                tag: str) -> None:
    """Alternative benchmarks from local data/ files (downloaded once):
    math500 (symbolic verify), mmlu (letter choice), mbpp (executable tests),
    lambada (teacher-forced final-word accuracy — a runtime control, near-free).
    """
    rng = np.random.default_rng(1234)
    data = Path(__file__).parent.parent / "data"
    results = {}

    if "math500" in benches:
        from math_verify import parse as mv_parse, verify as mv_verify

        rows = [json.loads(l) for l in open(data / "math500.jsonl")]
        by_level: dict[int, list] = {}
        for r in rows:
            by_level.setdefault(r["level"], []).append(r)
        picked = []
        for lvl in sorted(by_level):   # estratificado por nivel
            sub = by_level[lvl]
            take = max(1, round(n_bench * len(sub) / len(rows)))
            picked += [sub[i] for i in rng.permutation(len(sub))[:take]]
        picked = picked[:n_bench]
        ok = 0
        for j, r in enumerate(picked):
            out = _ask(model, tokenizer, rt,
                       r["problem"] + "\n\nSolve step by step and put your final "
                       "answer inside \\boxed{}.", 1024)
            m = re.findall(r"\\boxed\{", out)
            got = None
            if m:
                i = out.rfind("\\boxed{")
                depth, k = 0, i + 7
                while k < len(out):
                    depth += out[k] == "{"
                    if out[k] == "}":
                        if depth == 0:
                            break
                        depth -= 1
                    k += 1
                got = out[i:k + 1]
            try:
                ok += int(got is not None and
                          mv_verify(mv_parse("\\boxed{" + r["answer"] + "}"),
                                    mv_parse(got)))
            except Exception:
                pass
            if (j + 1) % 10 == 0:
                print(f"  math500 {j + 1}/{len(picked)}: {ok} ok")
        results["math500"] = ok / len(picked)
        print(f"[math500] {ok}/{len(picked)} = {ok / len(picked):.0%}")

    if "mmlu" in benches:
        rows = [json.loads(l) for l in open(data / "mmlu500.jsonl")][:n_bench if n_bench < 500 else 500]
        ok = 0
        for j, r in enumerate(rows):
            letters = "ABCD"
            q = r["question"] + "\n" + "\n".join(
                f"{letters[i]}. {c}" for i, c in enumerate(r["choices"]))
            out = _ask(model, tokenizer, rt,
                       q + "\n\nReply with ONLY the letter of the correct answer.", 8)
            m = re.search(r"\b([ABCD])\b", out)
            ok += int(bool(m) and letters.index(m.group(1)) == r["answer"])
            if (j + 1) % 50 == 0:
                print(f"  mmlu {j + 1}/{len(rows)}: {ok} ok")
        results["mmlu"] = ok / len(rows)
        print(f"[mmlu] {ok}/{len(rows)} = {ok / len(rows):.0%}")

    if "mbpp" in benches:
        rows = [json.loads(l) for l in open(data / "mbpp.jsonl")]
        rows = [rows[i] for i in rng.permutation(len(rows))[:n_bench]]
        ok = 0
        for j, r in enumerate(rows):
            prompt = (r["prompt"] + "\n\nYour function must satisfy this test:\n"
                      + r["test_list"][0] + "\n\nReply with ONLY the complete "
                      "Python function inside a ```python code block.")
            out = _ask(model, tokenizer, rt, prompt, 512)
            m = re.findall(r"```(?:python)?\n(.*?)```", out, re.DOTALL)
            code = m[0] if m else out
            program = ("\n".join(r.get("test_imports") or []) + "\n" + code
                       + "\n\n" + "\n".join(r["test_list"]) + "\nprint('PASS')\n")
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(program)
            try:
                res = subprocess.run(["python3", f.name], capture_output=True,
                                     text=True, timeout=15)
                ok += int("PASS" in res.stdout)
            except Exception:
                pass
            if (j + 1) % 10 == 0:
                print(f"  mbpp {j + 1}/{len(rows)}: {ok} ok")
        results["mbpp"] = ok / len(rows)
        print(f"[mbpp] {ok}/{len(rows)} = {ok / len(rows):.0%}")

    if "lambada" in benches and OPENAI_BASE:
        print("[lambada] omitido: teacher-forced no disponible vía API de chat")
    elif "lambada" in benches:
        rows = [json.loads(l) for l in open(data / "lambada.jsonl")]
        rows = [rows[i] for i in rng.permutation(len(rows))[:n_bench]]
        ok = 0
        for j, r in enumerate(rows):
            text = r["text"]
            prefix, last = text.rsplit(" ", 1)
            ids_full = tokenizer.encode(text)
            ids_pre = tokenizer.encode(prefix)
            tail = ids_full[len(ids_pre):]
            if not tail:
                continue
            out = model(mx.array(ids_full)[None])
            pred = np.array(mx.argmax(out[0, len(ids_pre) - 1:len(ids_full) - 1], axis=-1))
            mx.eval(out)
            ok += int((pred == np.array(tail)).all())
            del out
            rt.refresh_all()
            mx.clear_cache()
            if (j + 1) % 100 == 0:
                print(f"  lambada {j + 1}/{len(rows)}: {ok} ok")
        results["lambada"] = ok / len(rows)
        print(f"[lambada] {ok}/{len(rows)} = {ok / len(rows):.0%}")

    results["peak_gb"] = mx.get_peak_memory() / 1e9
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/bench_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[mem] peak {results['peak_gb']:.2f} GB")


def stage_kld(model, tokenizer, rt, data: str, chunks: int, chunk_len: int,
              ref: str | None, out: str | None, tag: str,
              decode: bool = False, refresh_every: int = 256) -> None:
    """KLD servido-vs-base por token (Accuracy is Not All You Need,
    2407.09141: la PPL media cancela daño por token; KLD no). Dos pasadas:
    con --kld-out guarda log-probs de la referencia (modo exact = la base);
    con --kld-ref carga y reporta KL(base ‖ servido) media/p95/p99."""
    rows = [json.loads(l) for l in open(data)][:chunks]   # jsonl con 'text'
    all_lp = []
    step = 0
    for row in rows:
        toks = tokenizer.encode(row["text"])[:chunk_len]
        if not decode:
            # Teacher-forced en UN forward (T>1): el prefill exacto lo sirve
            # bit-exacto por diseño → mide la exactitud del prefill, no el pool.
            ids = mx.array(toks)[None]
            z = model(ids)[0, :-1].astype(mx.float32)
            lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
            mx.eval(lp)
            all_lp.append(np.array(lp, dtype=np.float16))
            del z, lp
        else:
            # RÉGIMEN DE DECODE: prefijo corto exacto + resto token a token con
            # caché KV bajo la política del pool (refresh cada 256 como en
            # producción). Esta es la KLD que mide el peaje real del pool.
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
            k0 = max(1, min(16, len(toks) // 2))
            pf = model(mx.array(toks[:k0])[None], cache=cache)   # NO pisar `out`
            mx.eval(pf)
            del pf
            rt.refresh_all()                    # el pool aprende del prompt
            lps = []
            for t in range(k0, len(toks) - 1):
                z = model(mx.array([toks[t]])[None], cache=cache)[0, -1].astype(mx.float32)
                lp = z - mx.logsumexp(z)
                mx.eval(lp)
                lps.append(np.array(lp, dtype=np.float16))
                step += 1
                if step % refresh_every == 0:
                    rt.refresh_all()
            if lps:
                all_lp.append(np.stack(lps))
            del cache
        rt.refresh_all()
        mx.clear_cache()
    if out:
        np.savez_compressed(out, *all_lp)
        print(f"[kld] referencia guardada: {out} ({len(all_lp)} chunks)")
        return
    refz = np.load(ref)
    kls = []
    for i, lq in enumerate(all_lp):
        lp = refz[f"arr_{i}"].astype(np.float32)      # base
        lq = lq.astype(np.float32)                    # servido
        n = min(len(lp), len(lq))
        kl = (np.exp(lp[:n]) * (lp[:n] - lq[:n])).sum(-1)   # KL(P‖Q) por token
        kls.append(kl)
    kl = np.concatenate(kls)
    # curva por posición (dentro del chunk): ¿falla desde el principio o se
    # acumula con la longitud? Guardar el vector por chunk y la media por tramos.
    np.savez_compressed(f"runs/kld_{tag}_curve.npz", *kls)
    L = min(len(k) for k in kls)
    stack = np.stack([k[:L] for k in kls])
    tramos = {f"pos_{a}-{b}": float(stack[:, a:b].mean())
              for a, b in ((0, 64), (64, 128), (128, 256), (256, 448)) if b <= L}
    print("[kld] por tramo de posición:", {k: round(v, 4) for k, v in tramos.items()})
    res = {"kld_mean": float(kl.mean()), "kld_p95": float(np.percentile(kl, 95)),
           "by_position": tramos,
           "kld_p99": float(np.percentile(kl, 99)), "kld_max": float(kl.max()),
           "tokens": int(len(kl)), "peak_gb": mx.get_peak_memory() / 1e9}
    Path(f"runs/kld_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[kld] media {res['kld_mean']:.5f} · p95 {res['kld_p95']:.5f} · "
          f"p99 {res['kld_p99']:.5f} · max {res['kld_max']:.3f} ({len(kl)} tokens)")


def stage_traj(model, tokenizer, rt, n_prompts: int, gen_len: int,
               ref: str | None, out: str | None, tag: str) -> None:
    """Divergencia de trayectoria greedy (estilo Divergence-300@32 de
    Unsloth): mismos prompts, decodificación greedy, ¿cuándo divergen las
    trayectorias del servido respecto a la base exacta?"""
    data = Path(__file__).parent.parent / "data"
    rng = np.random.default_rng(1234)
    prompts = []
    for fname, key, k in (("math500.jsonl", "problem", n_prompts // 3),
                          ("mmlu500.jsonl", "question", n_prompts // 3),
                          ("mbpp.jsonl", "prompt", n_prompts - 2 * (n_prompts // 3))):
        rows = [json.loads(l) for l in open(data / fname)]
        prompts += [rows[i][key] for i in rng.permutation(len(rows))[:k]]
    trajs = []
    for j, p in enumerate(prompts):
        out_text_ids = []
        msgs = [{"role": "user", "content": p}]
        try:
            text = tokenizer.apply_chat_template(msgs, enable_thinking=False,
                                                 add_generation_prompt=True, tokenize=False)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=False)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        txt = generate(model, tokenizer, text, max_tokens=gen_len,
                       sampler=make_sampler(temp=0.0))
        out_text_ids = tokenizer.encode(txt)[:gen_len]
        rt.refresh_all()
        trajs.append(out_text_ids)
        if (j + 1) % 25 == 0:
            print(f"  traj {j + 1}/{len(prompts)}")
    if out:
        np.savez_compressed(out, *[np.array(t, dtype=np.int64) for t in trajs])
        print(f"[traj] referencia guardada: {out} ({len(trajs)} prompts)")
        return
    refz = np.load(ref)
    exact, first_div = 0, []
    for i, t in enumerate(trajs):
        r = refz[f"arr_{i}"].tolist()
        n = min(len(r), len(t))
        div = next((k for k in range(n) if r[k] != t[k]), None)
        if div is None and len(r) == len(t):
            exact += 1
        first_div.append(div if div is not None else n)
    res = {"prompts": len(trajs), "gen_len": gen_len,
           "exact_match": exact / len(trajs),
           "mean_first_divergence": float(np.mean(first_div)),
           "peak_gb": mx.get_peak_memory() / 1e9}
    Path(f"runs/traj_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[traj] exact-match@{gen_len}: {exact}/{len(trajs)} = "
          f"{exact / len(trajs):.0%} · divergencia media en token "
          f"{np.mean(first_div):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality accounting for Stream artifacts")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--stage", required=True,
                        choices=["ppl", "tasks", "bench", "kld", "traj"])
    parser.add_argument("--kld-out", default=None,
                        help="guardar log-probs/trayectorias de referencia aquí")
    parser.add_argument("--kld-ref", default=None,
                        help="comparar contra la referencia guardada")
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--kld-decode", action="store_true",
                        help="KLD en régimen de decode (token a token con caché)")
    parser.add_argument("--kld-refresh", type=int, default=256,
                        help="cadencia de refresh en decode-KLD (remedio del arranque)")
    parser.add_argument("--openai-base", default=None,
                        help="baseline externo OpenAI-compatible (p.ej. http://127.0.0.1:8080)")
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
    parser.add_argument("--bench", default="math500,mmlu,mbpp,lambada",
                        help="comma list: math500,mmlu,mbpp,lambada")
    parser.add_argument("--n", type=int, default=50,
                        help="items per bench (mmlu siempre usa hasta 500)")
    parser.add_argument("--centroid", default="uniform",
                        choices=["uniform", "empirical"])
    parser.add_argument("--p1-frac", default="1.0",
                        help="fracción saliente del P1 en el pool: '0.5' "
                             "uniforme o 'gate,up,down' (p.ej. '0.5,0.5,1.0')")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    set_seeds(1234)
    if args.openai_base:
        # Baseline externo: sin modelo MLX; el stage bench habla con la API.
        global OPENAI_BASE
        OPENAI_BASE = args.openai_base

        class _NoRT:
            @staticmethod
            def refresh_all():
                return 0

        stage_bench(None, None, _NoRT, args.bench.split(","), args.n, args.tag)
        return
    model, tokenizer, rt = load_runtime(args.artifact, args.pool_c, args.pool_k,
                                        args.serve_mode, args.orders, args.floor,
                                        args.centroid, args.p1_frac)
    print(f"[load] {mx.get_active_memory() / 1e9:.2f} GB")
    if args.stage == "ppl":
        stage_ppl(model, tokenizer, rt, args.data_code, args.data_general,
                  (args.chunks_code, args.chunks_general), args.chunk_len, args.tag)
    elif args.stage == "bench":
        stage_bench(model, tokenizer, rt, args.bench.split(","), args.n, args.tag)
    elif args.stage == "kld":
        stage_kld(model, tokenizer, rt, args.data_general,
                  args.chunks_general, args.chunk_len,
                  args.kld_ref, args.kld_out, args.tag, args.kld_decode,
                  args.kld_refresh)
    elif args.stage == "traj":
        stage_traj(model, tokenizer, rt, args.n, args.gen_len,
                   args.kld_ref, args.kld_out, args.tag)
    else:
        stage_tasks(model, tokenizer, rt, args.humaneval, args.gsm, args.tag)


if __name__ == "__main__":
    main()
