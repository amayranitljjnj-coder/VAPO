# BiPS 7B reproduction track

This track reproduces the existing BiPS optimization design with
Qwen2.5-VL-7B-Instruct and the earlier `BiPS_ECD_3B_filtered_v3` dataset. The
Counterfactual RLVR experiment uses a separate 7B-filtered IDK dataset. This
track keeps the BiPS actor-loss implementation
and official accuracy-only reward separate from the new detached nonlinear
reward implementation.

See `docs/BIPS_7B_REPRODUCTION.md` for the exact objectives and commands.
