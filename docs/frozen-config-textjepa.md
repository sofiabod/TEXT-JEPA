# Frozen Configuration — Text-JEPA

Date frozen: [fill in before first eval run]

All hyperparameters are frozen from experiments/configs/default.yaml.
No hyperparameter may be changed after this date if any eval result has been observed.

| Parameter | Value | Source |
|---|---|---|
| encoder_backbone | meta-llama/Llama-3.2-1B | agent-context.md spec |
| latent_dim | 256 | agent-context.md spec |
| predictor_layers | 2 | agent-context.md spec |
| predictor_heads | 4 | agent-context.md spec |
| predictor_mlp_ratio | 2 | agent-context.md spec |
| ema_momentum_start | 0.996 | agent-context.md spec |
| ema_momentum_end | 1.0 | agent-context.md spec |
| temporal_stride | 1 | level-1 flat; change only via config |
| context_window_k | 4 | agent-context.md spec |
| lr | 3e-4 | agent-context.md spec |
| weight_decay | 0.01 | agent-context.md spec |
| batch_size | 16 | agent-context.md spec |
| max_epochs | 200 | agent-context.md spec |
| early_stopping_patience | 30 | agent-context.md spec |
| bcs_num_slices | 1024 | hier_cost_exp BCS defaults |
| bcs_lmbd | 0.1 | hier_cost_exp BCS defaults |
| lambda_reg | [fill in] | validated on val loss before eval |

Seeds: 0, 1, 2, 3, 4
Train/val/test split: 70% / 10% / 20% by sequence position (temporal, no shuffle).
