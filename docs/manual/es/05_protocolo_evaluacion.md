# 5. Protocolo de evaluación

## 5.1 Determinismo

- **Greedy en todo**: `make_sampler(temp=0.0)` en las tareas y trayectorias;
  `temperature 0` en las llamadas al rival por API (en llama.cpp es greedy
  explícito; `repeat_penalty` por defecto 1.0 = apagado). Sin top-p, sin
  top-k, sin muestreo en ningún sitio.
- **Semillas**: `set_seeds(1234)`; con greedy solo afectan a qué ítems entran
  en cada muestra (`np.random.default_rng(1234).permutation`).
- **Pensamiento desactivado** (`enable_thinking=False` en la plantilla de
  chat); el rival es el checkpoint Instruct (sin modo de pensar).
- Consecuencia: cada cifra es repetible; la única fuente de variación entre
  runs es el estado del pool, que depende del orden de ítems (fijo).
- Misma máquina, mismos prompts, mismos parsers para todas las
  configuraciones y para el rival.

## 5.2 `eval_stream.py` — stages

```
python src/eval_stream.py --artifact <art> --stage <stage> [--serve-mode ...]
    [--pool-c C --pool-k K --orders ...] [--gen-refresh N] [--tag T] ...
```

| Stage | Qué mide | Salida |
|---|---|---|
| `ppl` | PPL teacher-forced por modo de servicio (código / general; `--chunks-* --chunk-len`) | `runs/ppl_<tag>.json` |
| `tasks` | HumanEval (`openai/openai_humaneval`) y GSM8K (`openai/gsm8k`) vía `datasets`; `--humaneval N --gsm N` | `runs/tasks_<tag>.json` |
| `bench` | MATH-500, MMLU, MBPP, LAMBADA desde `data/` congelado; `--bench lista --n N`; guarda `per_item` | `runs/bench_<tag>.json` |
| `kld` | KL(base ‖ servido) por token. Dos pasadas: `--kld-out` guarda los log-probs de la referencia (modo `exact`); `--kld-ref` compara. `--kld-decode` para el régimen de decode; `--kld-refresh N` | `runs/kld_<tag>.json`, `runs/kld_<tag>_curve.npz` |
| `traj` | Divergencia de trayectoria greedy (estilo Divergence-300@32): mismos prompts, 32 tokens, ¿en qué token divergen? | `runs/traj_<tag>.json` |
| `kldremote` | KLD de un baseline externo (llama-server `/completion`, `n_probs`) contra **nuestra** referencia guardada; KL truncada al top-100 de la referencia | `runs/kld_<tag>.json` |

Con `--openai-base URL` las tareas se envían a un servidor OpenAI-compatible
(el rival) con los mismos prompts y parsers; LAMBADA se omite (necesita
teacher-forcing).

## 5.3 Datasets congelados (`data/`)

| Fichero | Ítems | Muestreo en `bench` | Verificación |
|---|---|---|---|
| `math500.jsonl` | 500 | estratificado por nivel, `n` | `math_verify` simbólico sobre el último `\boxed{}` (parser con balance de llaves) |
| `mbpp.jsonl` | 257 (sanitized) | permutación, `n` | ejecución con `test_list` + `test_imports`, timeout 15 s, `PASS` |
| `mmlu500.jsonl` | 500 | primeros 500 (estratificado por materia al congelar) | letra `[ABCD]` en la respuesta (`max_tokens 8`) |
| `lambada.jsonl` | 5153 | permutación, `n` | teacher-forced: argmax de todos los tokens de la última palabra |
| `ifeval.jsonl` | 541 | — | sin usar |

Prompts: MATH "Solve step by step and put your final answer inside
\boxed{}" (1024 tokens); MBPP "Your function must satisfy this test: …
Reply with ONLY the complete Python function inside a ```python code block"
(512); HumanEval análogo (512); GSM8K "Reason briefly and end with the final
numeric answer" (384, último número).

**Refresh intra-generación**: `_ask` llama a `refresh_all()` cada
`GEN_REFRESH` tokens (y en la ráfaga si se activa) — hallazgo del arnés: las
primeras baterías corrían con el pool congelado durante cada respuesta
(~500 tokens). Las cifras de producción usan `--gen-refresh 128`.

## 5.4 KLD: tres regímenes

1. **Teacher-forced en un forward (T>1)**: mide la exactitud del prefill.
   Con prefill exacto es 0.000 por construcción (2278 tokens) — identidad
   bit a bit. No mide el pool.
2. **Régimen de decode** (`--kld-decode`): prefijo exacto de `k0 = 16`
   tokens con caché KV (`make_prompt_cache`), `refresh_all()` tras el
   prefijo, y el resto **token a token** alimentando el token verdadero
   (teacher-forced, pero con el estado del pool evolucionando como en
   decode), refresh cada `--kld-refresh`. Guarda el vector por posición y
   reporta media/p95/p99/max y la media por tramos 0–64 / 64–128 / 128–256 /
   256–448 (contados desde el prefijo). **Esta es la KLD del peaje del pool.**
3. **Rival remoto** (`kldremote`): mismos tokens por `/completion` con
   `cache_prompt` (cada paso evalúa un token: 1976 posiciones en 2.5 min en
   CPU), `n_probs 100`, probabilidades pre-muestreo (`top_logprobs`). KL
   sobre el top-100 de la referencia renormalizado en ambos: aproximación
   estándar que **subestima** frente a la KL completa; se declara. Posiciones
   en las que llama-server no devuelve probabilidades (token UTF-8 parcial
   en el borde del prompt → respuesta sin `completion_probabilities` o HTTP
   500) se saltan y se cuentan (`skipped_eos`: 4 de 1980).

Referencias guardadas: `runs/kld80_base_long.npz` (80B base exacta, wiki
4×512, 495 posiciones por chunk), `runs/kld30_base*.npz` (30B-Stream vs su
taper). Corpus: `nanite-moe/data/calib_general_qwen3/held_out.jsonl`
(held-out, nunca usado para `orders`).

## 5.5 Flips y estadística

- `per_item` por benchmark en cada `bench_*.json`; `flips(tag_a, tag_b)`
  cuenta correcto→incorrecto e incorrecto→correcto y calcula McNemar exacto
  (*Accuracy is Not All You Need*, 2407.09141).
- Potencia: a n=100 el IC 95 % es ±9.4 puntos en torno al 65 %; a n=300,
  ±5.4; a n=500, ±4.2. Pareado (mismos ítems) con ~20 % de discordantes: error
  típico de la diferencia ≈ 2 puntos a n=500 → una brecha de 5 puntos sale
  significativa; a n=300 queda en el borde. MBPP tiene techo en 257.
- Las igualdades de PPL (cuatro decimales sobre decenas de miles de tokens)
  son los resultados estadísticamente fuertes; las tareas a n≤100 soportan
  "indistinguible" y efectos grandes (−20 de la rotura de cobertura), no
  rankings finos.

## 5.6 Velocidad

- `examples/speed_nativo.py`: un proceso por medición (evita contaminación de
  la caché de Metal), rondas intercaladas, solo decode (el primer token no
  cuenta), 256 o 1024 tokens, mediana.
- `examples/citable_bench.sh`: tras **reinicio limpio**, 3 rondas de 1024
  tokens intercaladas; swap debe ser 0 (`sysctl vm.swapusage`) o el run se
  descarta. Citable: 35B 47.2 tok/s en frío; 80B caliente 17.3–21.5 (sus
  rondas en frío dispararon 1.1 GB de swap y se descartaron).
- Rival: `llama-bench`/`llama-server` con `-ngl 0` (único modo que arranca
  en 24 GB), `tg128`: 12.6 ± 3.7 tok/s.

## 5.7 El duelo (protocolo)

`examples/duel_udq2_v3.sh`: `llama-server -m <UD-Q2_K_XL.gguf> -ngl 0 -c 4096`,
espera `/health`, humo con una petición de chat, y `eval_stream --stage bench
--openai-base http://127.0.0.1:8080` con los mismos `n` y semilla. v1
(`-ngl 99`) murió por OOM de Metal; v2 (`--cpu-moe`) murió tras cargar. En
serie con lo nuestro: sus 30 GB paginados más nuestros 17 GB no caben juntos.

Rival: GGUF de Unsloth de linaje Dynamic 2.0 (según su card; el texto de Dynamic 3.0 es la referencia metodológica), https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF (`-ngl 0`, b10520).

## 5.8 Tests (`pytest -q`, sin modelos)

`test_unpack_pack_roundtrip`, `test_unpack_bits_order` (empaquetado);
`test_plane_reconstruction_exact_through_kernel` (P0+P1 == 4-bit a través de
`gather_qmm`); `test_cold_centroid_bias` (fold 1.5·s);
`test_anchored_q8_truncation_recovers_q4`; `test_pool_init_and_lookup`,
`test_pool_refresh_incremental_and_churn_cap`, `test_pool_p1_subset_invariant`
(máquina de estados del pool con memmaps sintéticos; K ⊆ C siempre);
`test_governor_thresholds`; `test_model_card_placeholder_gate`. CI fijado a
las versiones probadas.
