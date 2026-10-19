# Closed-loop agent parameter recovery — child 4-block task

This pipeline **simulates new foraging trajectories**. Each agent receives planet rewards, chooses stay/leave, visits new planets when it leaves, and updates its CRP particle-filter beliefs as it experiences the simulated environment. It then fits **both** existing alpha models to each generated trajectory, using the unchanged `fit_alpha_models.py` optimizer.

## Setup

Put the files here (the directory name can be `modeling` or `model_fits`):

```text
modeling/
  fit_alpha_models.py
  replication_choice_fit_child.jl
  particle_filter_child.jl
  run_full_agent_recovery.py               # NEW
  plot_full_agent_recovery.ipynb             # NEW
  data/new_pilot_data_child.csv
  full_agent_recovery/pilot_v1/              # CREATED
```

`fit_alpha_models.py` and the Julia files should be **the versions used for your empirical child model fitting**. You do not need `new_pilot_avg_child.csv` to run recovery.

Install Python packages: `pip install numpy pandas scipy matplotlib seaborn`.

Install Julia packages (once):

```powershell
julia -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "Distributions", "StatsBase", "Sobol"])'
```

Then edit the clearly marked settings near the top of `run_full_agent_recovery.py` and run:

```powershell
python run_full_agent_recovery.py
```

Start with `N_AGENTS_PER_MODEL = 2`, `WORKERS = 1`, and `N_SOBOL = 256` to test the whole pipeline. For a final run, increase to 50–100 agents **per generating model**, `N_SOBOL = 1024`, `N_LOCAL_STARTS = 5`. Fitting is substantially slower than the previous conditional-choice recovery because **every simulated trajectory needs independent Julia latent extraction for both fitted models**.

## Model variants

| Label | Fixed CRP α | γ_coef | β | Free fitted parameters |
|---|---:|---|---:|---|
| `alpha_0` | 0.0 | 0.0 | 1.0 | γ_base, ε |
| `alpha_1` | 0.2 | Fitted | 1.0 | γ_base, γ_coef, ε |

`alpha_1` is **not** CRP α = 1. The original child-model **positive** `gamma_coef` sign is preserved: `gamma_t = sigmoid(gamma_base + gamma_coef*U_t)`.

True parameter sampling ranges can be **different** from fitting bounds and are controlled by `TRUE_PARAMETER_RANGES` in the new script. They must fit inside `fm.BOUNDS` in `fit_alpha_models.py`. If you want to test recovery across the entire fitting space, set the true ranges equal to those bounds. This is a harder test than a narrower, behaviorally plausible range.

## Environment (why no `exp_struc.csv` is needed)

The original parameter-recovery simulator specified:

- 4 blocks, each capped at **360 seconds** or **20 visited planets**.
- Initial planet reward: rounded, clipped draw from `Normal(100, 5)`.
- Reward decay per subsequent harvest: `Beta(13,51)` for poor, `Beta(50,50)` for neutral, `Beta(50,12)` for rich; next reward is rounded `previous_reward * decay`.
- Harvest + ITI = **4.5 seconds**; travel + alien time = **15.5 seconds**; initial elapsed time per block = **5.5 seconds**.
- Cumulative overall reward rate is maintained across blocks, matching the current empirical choice model.

**Unavailable original schedule:** The uploaded files don't contain the original `exp_struc.csv` specifying the predetermined sequence of planet types. To keep the simulated task close to the children's actual four-block conditions, this script estimates **block-specific probabilities** of poor/neutral/rich planets from **distinct planet visits** in `new_pilot_data_child.csv`, then independently samples each next planet's type from the corresponding block distribution. The sampled block proportions are saved in `environment_probabilities.csv`. This is a **stochastic reconstruction**, not an exact replay of the original scheduled galaxy order. Substitute the authentic schedule in the Julia simulator if exact schedule replication is essential.

Each choice genuinely changes the trajectory. The final reward decision in each block is censored (`observed_choice` missing), matching the child empirical preprocessing; its actual generated action remains in `simulated_choice`. This can yield variable numbers of harvested rewards, planets, and uncensored decisions across agents.

## Important modeling and stochasticity distinctions

- The generator calls `cluster_posteriors`, `predictive_samples`, and `update_particles` from your **current** `replication_choice_fit_child.jl`; it does not use the buggy double-ε action sampling of the archived `gen_crp_softmax.jl`.
- The generator and the fitter use **independent random seeds**, with equal default particle/sample counts. The fitted CRP latents are freshly recomputed on each simulated trajectory for each candidate model. They are **not** reused from the generative run or from empirical children.
- The likelihood is the same trial-level choice likelihood and Sobol + L-BFGS-B optimization as `fit_alpha_models.py` (rather than the archive's Monte Carlo-averaged likelihood).
- **Caveat in inherited model:** The current `log_prior_particle` counts all visited planets in `N`, including ones producing no decay observations, and `normalize_logweights` uses uniform fallback if every prior is zero. At `alpha = 0`, this could permit unintended new-cluster posterior mass after zero-decay planets. The simulator deliberately shares this code with the empirical fitter, but the alpha-0 semantics should be independently audited before scientific conclusions.
- There is no separate validation of the Julia numerical execution in this environment (Julia is not installed here). Python syntax, environment extraction, and summary/visualization logic have been tested. Test one agent per model first.

## Outputs

For `RUN_NAME = "pilot_v1"`:

```text
full_agent_recovery/pilot_v1/
  recovery_config.json                 # source hashes + settings
  environment_probabilities.csv
  recovery_all_fits.csv                # 2 rows/agent (both fitted models)
  recovery_matched.csv                 # 1 row/agent (matched fitting model)
  model_recovery.csv                   # AIC/BIC comparisons and classification
  failures.csv
  runs/gen_alpha_0_agent_0001/
    simulated_trials.csv               # FULL new closed-loop reward/choice trajectory
    latents_alpha_0.csv
    latents_alpha_1.csv
    fits.csv
  runs/gen_alpha_1_agent_0001/...
  figures/                              # created by visualization notebook
```

**Safe reruns:** The pipeline skips agents that already have both completed fits. You can increase `N_AGENTS_PER_MODEL` with the same `RUN_NAME` without deleting files, because each agent's parameter draws and simulation seed are independent of total sample size. If **any other model, parameter range, environment or optimization setting changes**, use a **new `RUN_NAME`**; the configuration check will protect previous results. Do not manually delete the individual run folders while a fitting job is active, particularly when OneDrive is syncing.

Open `plot_full_agent_recovery.ipynb` from the `modeling` folder. Its first cell sets `RUN_NAME` and produces true-vs-fitted parameter plots, AIC/BIC model-classification confusion matrices, and recovery summary statistics.
