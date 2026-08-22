"""Máster 4-bit protegido por saliencia (AWQ nativo de mlx-lm) para MoE Qwen3.

La dirección D del roadmap en nuestro estilo: AWQ escala los canales salientes
ANTES de cuantizar y pliega la inversa en la capa previa — el resultado sigue
siendo afín plano por grupo → el split en planos (P0/P1) y `gather_qmm` se
conservan, y el suelo P0 hereda la protección (deja de ser el peor 2-bit).

mlx-lm trae AWQ con soporte MoE (deepseek_v2) pero no registra `qwen3_moe`:
aquí se añade (atención tipo llama + switch_mlp sin shared expert).

Usage (desde un BF16):
    python src/awq_master.py --model Qwen/Qwen3-30B-A3B --bits 4 --group-size 64 \
        --mlx-path models/qwen3-30b-awq4 --num-samples 32 --sequence-length 512
"""

from __future__ import annotations

import sys

from mlx_lm.quant import awq


def register_qwen3_moe() -> None:
    awq.AWQ_MODEL_CONFIGS["qwen3_moe"] = awq.AWQConfig(
        embed="embed_tokens",
        lm_head="lm_head",
        no_clip=["q_proj", "k_proj"],
        scale_configs=[
            awq.ScaleConfig(block="self_attn", prev="input_layernorm",
                            layers=["q_proj", "k_proj", "v_proj"], kwargs=["mask"]),
            # down_proj de los expertos: escala plegada en up_proj (por experto)
            awq.ScaleConfig(prev="mlp.switch_mlp.up_proj",
                            layers=["mlp.switch_mlp.down_proj"],
                            use_config=lambda block: "switch_mlp" in block.mlp,
                            kwargs=["indices"]),
            # gate/up de los expertos (+ el router, solo escalado): plegado en la norma
            awq.ScaleConfig(block="mlp", prev="post_attention_layernorm",
                            layers=["switch_mlp.gate_proj", "switch_mlp.up_proj", "gate"],
                            use_config=lambda block: "switch_mlp" in block.mlp),
            # capas densas (si las hubiera)
            awq.ScaleConfig(prev="mlp.up_proj", layers=["mlp.down_proj"],
                            use_config=lambda block: "switch_mlp" not in block.mlp),
            awq.ScaleConfig(block="mlp", prev="post_attention_layernorm",
                            layers=["gate_proj", "up_proj"],
                            use_config=lambda block: "switch_mlp" not in block.mlp),
        ],
    )


if __name__ == "__main__":
    register_qwen3_moe()
    sys.argv[0] = "mlx_lm.quant.awq"
    awq.main()
