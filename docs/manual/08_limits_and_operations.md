# 8. Declared limits and operational lessons

## 8.1 Limits of the method

- **Format**: flat affine quantization only (MLX). It does not port to GGUF
  K-quants or IQ-quants; upward (Q8_0 → 2×Q4_0) it does.
- **Generality**: one machine, one family (Qwen: 3, 3.5, Next). Qwen's decode
  routing is highly local; a load-balanced router (Mixtral, OLMoE) could
  defeat recency-based residency and must be measured before claiming the
  coverage laws beyond "MoE with high decode locality". By directive the
  program is Qwen-only; generality remains a limitation, not a task.
- **Prose fidelity**: the 80B at 24 GB is not competitive in general-text KLD
  (0.774 vs 0.195 for the static one); it is on tasks and as a system.
- **Coverage**: every routed slot needs at least a floor; naked drops of
  dominant slots collapse the model, not degrade it. The 80B has no viable
  universal floor at 24 GB.
- **Speculative decoding**: blocked on Qwen's hybrid generation in current
  mlx-lm (SSM caches not trimmable). An implementation limit, not a
  fundamental one (*The Mamba in the Llama*).
- **Statistical power**: tasks at n=15–100; only the PPL equalities and the
  KLDs (thousands of tokens) are strong. See §5.5.
- **Engineering surface**: a private mlx-lm helper (`_gather_sort`),
  `sorted_indices` semantics, patched constructors; pinned versions.
- **Citable speed**: the 80B only has a warm figure (cold rounds trigger
  swap).
- **Cold storage**: paging the floor from a USB disk was >4× slower (run
  abandoned). Slow tiers are for refresh-time reads, never on the hot path.

- **Duel baseline.** CPU-only is the only way the 30 GB GGUF runs on 24 GB,
  but the strongest Apple-native rival (flash-moe / Anemll, SSD-streamed
  experts, custom Metal, no miss-time floor) has not been dueled yet. A
  15/15 at n=15 only bounds the failure rate below ~20%.

## 8.2 Hard memory rules

1. Check free RAM before loading; abort with a clear message if there is no
   budget.
2. Log `mx.get_active_memory()` / `mx.get_peak_memory()` after each phase.
3. **Swap = 0 always**; if a benchmark triggers swapouts, the result is
   discarded and the run is marked invalid. Real incidents: the duel launched
   on top of f48 (11 GB of swap); OVF measured with 3–6 GB of swap (5.7 tok/s
   contaminated → clean re-measurement 5.8).
4. The 80B's pools fit up to C=290 (19.8 GB); C=340 (20.7 GB) does not.
   floor2d at C=240 (pool 16.5 + floor 6) thrashes.
5. The rival (30 GB paged) and our 80B (17 GB) never at the same time.

## 8.3 Unattended chains (`examples/*.sh`)

Pattern: `nohup` + `caffeinate` (+ Amphetamine), gate by **marker** in
`runs/ablation.log` (`until grep -q "X COMPLETO" … && ! pgrep -f
"eval_stream|llama-|src/serve.py"; do sleep 60; done`), smoke test of one
chunk before any multi-hour run, `OK`/`FALLO` marker per stage.

Lessons paid for:

- **Stale markers open gates** (three incidents: an old `F48 FALLO`
  launched the duel on top of f48; the previous day's `RIVAL-KLD COMPLETO`
  started the OVF early; a stale `objetivo_kld` waiter started llama-server
  during the OVF). Remedy: gates with marker counts (≥2) or dated markers;
  kill obsolete waiters when reconfiguring the queue.
- **A reboot kills every chain** without leaving a trace; detect it via
  `boottime` (`sysctl kern.boottime`, parsed with awk) and re-orchestrate
  (`morning_orchestrator.sh`).
- **Smoke test before hours**: a `KeyError` or a clobbered variable (`out`
  shadowing the output) threw away 46-minute runs. First chunk + check that
  the `.npz` exists.
- **llama-server**: `/health` ok does not guarantee probabilities; partial
  UTF-8 tokens at the prompt boundary return responses without
  `completion_probabilities` or HTTP 500 ("does not match the expected
  Content-only format"). The parser skips and counts.
- **Count the times properly**: with `cache_prompt` the rival evaluates one
  token per step (1976 positions in 2.5 min); without it, hours.

## 8.4 Large downloads (HF)

- `hf download` **serially**, never in parallel (they throttle each other);
  restarting a live download abandons the xet partials (74 GB of orphans
  purged); `du` lies during the download — measure with `netstat`/`lsof`.
- A hotspot ceiling of ~1 MB/s makes downloading 132 GB pointless; on good
  Wi-Fi ~14 MB/s.
- The 235B was rebuilt whole once for consuming the shards before writing
  the skeleton: now the skeleton goes first.

## 8.5 Disk

- Deletion decisions belong to the user. When an `rm -rf` of the 35B
  artifact was blocked, it was moved to the external disk and its sha256
  verified against HF.
- The external disk disconnects sometimes: nothing on the hot path can live
  there.
- Inventory and free space in §4.7.

## 8.6 What not to do (summary for the next person who touches this)

- Do not rebuild the 80B's full pool on every refresh (10×).
- Do not leave `inds` lazy in prefill under memory pressure (garbage indices).
- Do not evaluate with the pool frozen during generation (`--gen-refresh`).
- Do not compare speeds across different sessions (the "87 vs 104" was an
  artifact); interleaved rounds, one process per measurement.
- Do not cite figures with swap.
- Do not generalize subsampling recipes across models without measuring them.
