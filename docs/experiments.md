# Research experiments

The tutorials are the supported educational starting point. Research scripts
now live under `experiments/`, retaining their internal directory structure and
local imports. Paths in the table below are relative to the repository root. Run them **from
the directory containing the script**, with the repository virtual environment
activated. They can be expensive and may overwrite local generated results.

| Directory | Entry points | Dependencies beyond numerical core |
| --- | --- | --- |
| `experiments/van_der_pol/EVT` | `train_model.py`, `verify_evt_new.py`, `verify_ic.py` | None |
| `experiments/van_der_pol/NLF` | `train_vdp.py`, `plot_roa.py` | dReal for training/verification |
| `experiments/van_der_pol/ICNN` | `train_vdp.py`, `plot_roa.py` | dReal for training/verification |
| `experiments/van_der_pol/PINN` | `train_pinn.py`, `plot_zubov_roa.py` | dReal |
| `experiments/van_der_pol/SOS` | `verify.py` | Drake |
| `experiments/dim_scaling/EVT` | `lyapunov_core.py`, `lyapunov_core_new.py`, `lyapunov_core_softplus.py` | None |
| `experiments/dim_scaling/{NLF,ICNN,PINN}` | `train_scaling.py` | dReal |
| `experiments/dim_scaling/SOS` | `verify.py` | Drake |

For example, `cd experiments/van_der_pol/EVT && python train_model.py` trains the historical
Gram candidate and saves `models/lyapunov_model.pth`. Back up the tracked
checkpoint if you intend to retain it. Tutorial outputs use a separate ignored
directory and never overwrite research checkpoints.

## Variants and retained artifacts

`verify_evt.py` and `verify_evt_new.py` are different experiment versions, as are
the SOS variants. They have not been deduplicated because their sampling and
acceptance logic differ. The former `lyapunov_core_new copy.py` is now
`lyapunov_core_softplus.py`: it uses softplus-squared penalties and is not an
identical backup. Existing `.pth`, `.png`, and `.pdf` files moved with their experiment directories, preserving internal relative
paths. `plot_roa_comparison.py` resolves its checkpoints and output directory
relative to its own location, so it also works when launched from the repository
root. Ignored personal notes and artwork
were left untouched.

## Interpretation and known limitations

- `VanDerPol.vector_field` uses the **reverse-time** system
  `(-y, x + mu*(x*x - 1)*y)`, stable near the origin for positive `mu`.
  Setting `time_reverse=True` in the legacy EVT trainer negates this field again;
  it is not the setting used by the tutorials.
- The EVT Gram candidate uses ten features. Its default feature count and device
  buffer placement have been corrected, and `mu` now affects the dynamics.
  The feature map's `log(cosh)` evaluation avoids overflow. State-dictionary
  names and tensor shapes for the existing ten-feature checkpoint are unchanged.
- The historical NLF candidate `NN(x)^2 + a*||x||^2` does **not** enforce
  `V(0)=0` when the network has biases. Its symbolic conversion iterates child
  modules rather than reproducing the forward activation order, and its
  SAT/UNSAT print labels are reversed. Those baselines require a separate
  mathematical and solver audit before their verification claims can be used.
  Their architectures and saved checkpoints were preserved here.
- A shell `rho_core <= V <= rho` excludes the equilibrium. A full ROA claim
  requires a separate local argument and compatible invariant sets. Neither
  a negative finite sample maximum nor an EVT fit fills that gap.
- SGLD particles and pruning introduce dependence and sampling bias. A fitted
  GEV endpoint requires assumptions about coverage and the tail regime.
  Ordinary KS p-values after fitting on the same data are not calibrated
  goodness-of-fit tests; bootstrap endpoint percentiles also need validation.
- High-dimensional linear systems can be checked against a quadratic Lyapunov
  equation. Use even dimensions for the paired oscillator construction; odd
  dimensions in some versions leave an undamped zero mode.

The tutorials expose these distinctions, use independent rejection samples for
the EVT illustration, and keep adversarial search as a separate diagnostic.
They are intentionally smaller than the reflected-SGLD research workflow.
