# Tutorial guide

Install `requirements/tutorials.txt` from the repository root, then launch
`python -m jupyterlab tutorials`. Work through notebooks 01–03 in order. Each
starts from a fixed seed and can also run alone. Basic Python, derivatives,
matrices, and the idea of an ODE are sufficient prerequisites.

1. **Dynamics and Lyapunov functions**: visualize reverse-time Van der Pol,
   derive a local quadratic candidate, compare automatic and finite-difference
   derivatives, and inspect trajectories. About 5–10 minutes of reading.
2. **Train a candidate**: reuse the actual EVT feature map, learn a positive Gram
   matrix, compare held-out diagnostics, and visualize candidate sublevels.
   About 10–15 minutes of reading; 300 CPU training steps by default.
3. **EVT assessment**: assess a fresh candidate or optionally load notebook 02's
   checkpoint, sample a shell, search for violations, fit block maxima, and
   inspect bootstrap endpoint sensitivity. About 15–20 minutes of reading;
   the short bootstrap is a demonstration, not a calibrated confidence claim.

Execution time depends on hardware. All cells are offline after installation.
Figures display inline. Only notebook 02 writes a small checkpoint, under ignored
`outputs/`; notebook 03 defaults to an independent warm-started model so no
hidden execution order is required. Change its `USE_TRAINED_CHECKPOINT` flag to
connect the complete train → assess workflow.

To verify everything from any working directory:

```bash
/path/to/repo/.venv/bin/python /path/to/repo/scripts/run_tutorials.py
```

The runner uses the interpreter that launched it for its fresh kernels. Executed
copies with plots are saved in `tutorials/outputs/executed/`; source notebooks
stay output-free. Each notebook contains exercises and a short interpretation
of expected plots. A positive sampled derivative is a counterexample for that
candidate and domain; a negative maximum is only evidence at sampled points.
