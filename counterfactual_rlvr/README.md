# Counterfactual RLVR 7B training package

This directory contains the small, repository-tracked pieces used by the
two-stage Qwen2.5-VL-7B-Instruct experiment. The complete procedure, KL audit,
hardware assumptions, data contract, calibration rule, and checkpoint layout
are documented in `docs/COUNTERFACTUAL_RLVR_7B_TRAINING.md`.

The rule reward in `reward_accuracy.py` returns binary accuracy only. The
detached nonlinear KL multiplier is applied centrally by the patched VeRL
trainer so incorrect answers remain exactly zero.
