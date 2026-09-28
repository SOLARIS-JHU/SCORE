# SCORE experiments

- [`van_der_pol/`](van_der_pol/): the two-dimensional nonlinear ROA benchmark,
  comparison baselines, trained checkpoints, and paper figures.
- [`dim_scaling/`](dim_scaling/): dimension-scaling benchmarks on dense linear
  systems, with EVT, NLF, ICNN, PINN, and SOS variants.

See the [experiment guide](../docs/experiments.md) for entry points, optional
solver dependencies, and interpretation notes. For the learning path, start with
[the tutorials](../tutorials/README.md).

Most historical scripts use local imports and paths. Activate the repository's
virtual environment, then run them from their containing directory, for example:

```bash
cd experiments/van_der_pol/EVT
python train_model.py
```

The comparison plot can also run directly from the repository root:

```bash
python experiments/van_der_pol/plot_roa_comparison.py
```

Training and plotting may overwrite the corresponding experiment's saved
checkpoint or figure. Tutorial outputs are stored separately in
`tutorials/outputs/`.
