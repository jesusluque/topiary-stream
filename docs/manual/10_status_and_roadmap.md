# 10. Status as of 2026-08-23 and roadmap

## 10.1 Status

- **Runtime**: production = 80B C=240 K=32 `nosync` refresh 128 (tasks) /
  256 (speed); 35B/30B resident-p0 K=32 with governor. Exact prefill in
  all of them. Tests green; CI pinned.
- **Closed evidence**: pyramid controls; 35B/80B badges; 5×4 bench;
  coverage law on the same model (C=120/240/290); P1 subsampling
  (8 points); 235B mapped (4 modes + f48); hostile metrics (TF KLD,
  decode, curves, trajectories); full duel against Unsloth including its
  KLD against our base; governor demonstrated; 35B cold boot.
- **"Beat the table" campaign**: cancelled on 23/08 by user decision after
  two rounds without return (cadence does not move tasks; 2-bit coverage
  makes them worse). Four fixes retired; seven levers implemented without
  measuring (not claimed).
- **Written**: paper with §2.7 (decode regime and duel), README, website,
  campaign report, roadmap, this manual. HF cards of the private repos
  without the duel note.
- **GPU queue**: empty.

## 10.2 Roadmap (summary; details in `paper/ROADMAP.md`)

**Pre-submission (pure measurement, no runtime changes):**

1. n≥300 per benchmark with paired bootstrap CI — full MATH-500 (500) and
   MBPP 257 on the 80B and the rival, serially, ~12 h GPU + ~17 h CPU. Turns
   "~5 points n.s." into significant or indistinguishable; does not change
   directions.
2. Vendor `_gather_sort` (one hour).
3. Swap-free cold boot of the 80B (freshly rebooted machine, nothing else
   loaded).
4. HF cards and public flip when the user decides.

**Open directions, by expected return under the rule "no clear return,
no go":**

- **E. 16 GB profile**: package 30B-Stream (9.2 GB) / 0:0:4 (10.4 GB) with
  a conservative config and an aggressive governor. Packaging and
  measurement only.
- **F. Exact mode as an evaluation service**: true PPL/KLD of unservable
  models at 4 GB peak, in batch.
- **D. Salience-protected master (affine AWQ/imatrix)**: the only untested
  lever for the 2-bit plane's quality residue; needs disk and GPU hours.
- **B. Damage-guided residency**: retain the experts whose miss hurts (the
  per-position KLD curves already exist to analyze it before coding).
- **C. Self-speculation with the floor as draft**: classic MoE only (235B);
  blocked on hybrids.
- **G. CUDA port on AWS g6**: Q8 → 2×Q4 pyramid on stock kernels
  (Marlin/machete) in vLLM, three-tier residency, replication of the
  coverage law; 1–2 weeks, $40–80. Upstream contributions: dynamic biases
  to FusedMoE; Q8_0→2×Q4_0 split to the llama.cpp RFC.

Retired: second family (OLMoE) by the Qwen-only directive; HOBBIT baseline
(omission justified in the paper); MLX-native AWQ parked.

## 10.3 How to keep this manual current

Every new result: a file in `runs/`, a row in §6, and if negative an entry
in §7; if a default changes, §2.7 and §10.1; if a command changes, §9. The
paper and the README are updated afterwards, not before.
