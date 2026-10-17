"""Child task CRP alpha-model comparison (beta fixed at 1).
Place this file, replication_choice_fit_child.jl and particle_filter_child.jl
under model_fits/. Input: data/new_pilot_data_child.csv.
Needs Julia + packages CSV, DataFrames, Distributions, StatsBase, Sobol;
Python pandas, numpy, scipy. Run as a script or `%run fit_alpha_models.py`.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc

# ====================== USER SETTINGS ======================
# Keep this script inside model_fits/; no hardcoded Windows path needed.
BASE = Path(__file__).resolve().parent
BATCH = "child"
MODEL = "both"                     # "alpha_0", "alpha_1", or "both"
JULIA = "julia"                    # Or full path to julia.exe
WORKERS = min(4, os.cpu_count() or 2)
N_SOBOL = 1024                    # Must be a power of 2
N_LOCAL_STARTS = 5
N_PARTICLES = 200
N_SAMPLES = 200
SEED = 42
REFIT = True                      # True: regenerate latents and refit
GAMMA_SIGN = +1                    # +1 matches archived child run_softmax_alpha_free.jl
TRIAL_CSV = BASE / "data" / "new_pilot_data_child.csv"
MODEL_TRIAL_CSV = BASE / "alpha_model_results" / "prepared_choices_child.csv"
QUESTIONNAIRE_CSV = BASE / "data" / "replication_questionnaires.csv"  # optional
SOURCE_JULIA = BASE / "replication_choice_fit_child.jl"
OUTPUT_ROOT = BASE / "alpha_model_results"
# ============================================================

# The original archive's model labels are NOT literal alpha=0 vs alpha=1.
# Its alpha_0 model has CRP alpha=0 and gamma_coef=0.
# Its alpha_1 model has CRP alpha=0.2 and gamma_coef free.
SPECS = {
    "alpha_0": {"alpha": 0.0, "free": ["gamma_base", "epsilon"]},
    "alpha_1": {"alpha": 0.2, "free": ["gamma_base", "gamma_coef", "epsilon"]},
}
BOUNDS = {"gamma_base": (-10.0, 10.0), "gamma_coef": (-3.0, 3.0), "epsilon": (0.0, 0.10)}

# This Julia helper extracts identical trial-level predictive quantities from
# observed_nll() but only ONCE because alpha is fixed for each model. The
# numerical optimizer runs in Python over the cached choice-level predictions.
JULIA_HELPER = r'''
using CSV, DataFrames, Random, Statistics
include(ARGS[1])

# Run the extraction in a function to avoid Julia soft-scope ambiguity.
function extract_latents()
sub_num = parse(Int, ARGS[2])
trial_csv = ARGS[3]
out_csv = ARGS[4]
alpha = parse(Float64, ARGS[5])
n_particles = parse(Int, ARGS[6])
n_samples = parse(Int, ARGS[7])
seed = parse(Int, ARGS[8])

planets, n_rewards, n_expected = prepare_subject(trial_csv, sub_num)
Random.seed!(seed + sub_num)
particles, weights = init_particle_filter(n_particles, 0.5, 0.25, 1.0, alpha)
total_reward, total_time, N = 0.0, 5.5, 0
out = DataFrame(choice=Int[], v_stay=Float64[], leave_scale=Float64[], U=Float64[])

for planet in planets
    rs = planet.rewards
    decays = Float64[]
    prev_reward = 0.0
    last_post = nothing
    for j in eachindex(rs)
        reward = rs[j]
        if j > firstindex(rs)
            push!(decays, prev_reward == 0 ? 1.0 : reward / prev_reward)
        end
        prev_reward = reward
        total_reward += reward
        total_time += 4.5
        post = cluster_posteriors(particles, N, alpha, decays)
        last_post = post
        mean_decay, U = predictive_samples(particles, weights, post, n_samples)
        choice = planet.choices[j]
        if !ismissing(choice)
            push!(out, (Int(choice), mean_decay * reward,
                        (total_reward / total_time) * 4.5, U))
        end
    end
    if !isempty(decays)
        particles, weights = update_particles(particles, weights, last_post, decays)
    end
    N += 1
    total_time += 15.5
end
nrow(out) == n_expected || error("Choice count mismatch")
mkpath(dirname(out_csv))
CSV.write(out_csv * ".tmp", out)
mv(out_csv * ".tmp", out_csv; force=true)
println("LATENTS sub=$sub_num alpha=$alpha choices=$(nrow(out)) rewards=$n_rewards")
end

extract_latents()
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    """Create observed choices from rewards when raw keypresses are unavailable.
    Last reward on the final planet of each block is censored (missing).
    """
    for f in (TRIAL_CSV, SOURCE_JULIA, SOURCE_JULIA.with_name("particle_filter_child.jl")):
        if not f.is_file():
            raise FileNotFoundError(f"Missing file: {f}")
    d = pd.read_csv(TRIAL_CSV)
    keys = ["sub_num", "block", "true_planet", "stay_num"]
    required = keys + ["reward"]
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError(f"Trial data lacks {sorted(missing)}")
    if d[required].isna().any().any() or d.duplicated(keys).any():
        raise ValueError("Missing or duplicated trial keys/rewards")
    d = d.sort_values(keys).reset_index(drop=True)
    # Every reward except a planet's final reward is followed by a stay.
    d["observed_choice"] = 1.0
    end_of_planet = ~d[["sub_num", "block", "true_planet"]].eq(
        d[["sub_num", "block", "true_planet"]].shift(-1)).all(axis=1)
    same_block_next = d[["sub_num", "block"]].eq(
        d[["sub_num", "block"]].shift(-1)).all(axis=1)
    d.loc[end_of_planet & same_block_next, "observed_choice"] = 0.0
    d.loc[end_of_planet & ~same_block_next, "observed_choice"] = np.nan
    # Check 0..n-1 stays on every planet.
    for key, g in d.groupby(["sub_num", "block", "true_planet"], sort=False):
        if g.stay_num.tolist() != list(range(len(g))):
            raise ValueError(f"Noncontiguous stay_num on planet {key}")
    MODEL_TRIAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(MODEL_TRIAL_CSV, index=False)
    n_censored = int(d.observed_choice.isna().sum())
    n_choices = int(d.observed_choice.notna().sum())
    print(f"Validated {len(d)} rewards, {n_choices} inferred choices, "
          f"{n_censored} censored block-final decisions")
    return sorted(int(x) for x in d.sub_num.unique()), n_choices

def fit_latents(path, spec, model, sub):
    z = pd.read_csv(path)
    if not len(z) or z[list(("choice", "v_stay", "leave_scale", "U"))].isna().any().any():
        raise ValueError(f"Missing or invalid latents for sub {sub}")
    if not np.isfinite(z[["v_stay", "leave_scale", "U"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite latent estimates for sub {sub}")
    choices = z["choice"].to_numpy(dtype=int)
    assert np.isin(choices, [0, 1]).all()
    vs, ls, U = [z[c].to_numpy(dtype=float) for c in ["v_stay", "leave_scale", "U"]]
    names = spec["free"]
    limits = np.array([BOUNDS[x] for x in names], dtype=float)
    low, high = limits[:, 0], limits[:, 1]

    def objective(theta):
        p = dict(zip(names, theta))
        gamma = expit(p["gamma_base"] + GAMMA_SIGN * p.get("gamma_coef", 0.0) * U)
        stay_prob = (1 - p["epsilon"]) * expit(vs - ls * gamma) + 0.5 * p["epsilon"]
        stay_prob = np.clip(stay_prob, 1e-12, 1 - 1e-12)
        return float(-np.where(choices == 1, np.log(stay_prob), np.log1p(-stay_prob)).sum())

    sobol = qmc.Sobol(d=len(names), scramble=True, seed=SEED + sub)
    points = low + sobol.random_base2(m=int(np.log2(N_SOBOL))) * (high - low)
    values = np.array([objective(x) for x in points])
    best_idxs = np.argsort(values)[:min(N_LOCAL_STARTS, N_SOBOL)]
    winners = []
    for idx in best_idxs:
        res = minimize(objective, points[idx], method="L-BFGS-B",
                       bounds=limits.tolist(), options={"maxiter": 300, "ftol": 1e-11})
        if np.isfinite(res.fun):
            winners.append(res)
    if not winners:
        raise RuntimeError(f"Optimization failed for sub {sub}, {model}")
    best = min(winners, key=lambda r: r.fun)
    pars = dict(zip(names, best.x))
    k = len(names)
    return {
        "sub_num": sub, "model": model, "alpha": spec["alpha"],
        "gamma_base": pars["gamma_base"], "gamma_coef": pars.get("gamma_coef", 0.0),
        "beta": 1.0, "epsilon": pars["epsilon"],
        "neg_llh": float(best.fun), "n_choices": len(z), "k": k,
        "AIC": 2*k + 2*best.fun, "BIC": k*np.log(len(z)) + 2*best.fun,
        "n_sobol": N_SOBOL, "n_local_starts": len(winners),
        "n_particles": N_PARTICLES, "n_samples": N_SAMPLES,
        "seed": SEED + sub, "optimizer_converged": bool(best.success),
    }


def run_subject(model, spec, sub, helper_path):
    folder = OUTPUT_ROOT / model
    latent_path = folder / "latents" / f"sub{sub}.csv"
    fit_path = folder / "subjects" / f"sub{sub}.csv"
    if REFIT or not latent_path.is_file():
        cmd = [str(JULIA), "--startup-file=no", str(helper_path), str(SOURCE_JULIA),
               str(sub), str(MODEL_TRIAL_CSV), str(latent_path), str(spec["alpha"]),
               str(N_PARTICLES), str(N_SAMPLES), str(SEED)]
        out = subprocess.run(cmd, text=True, capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(f"Julia failed sub {sub}:\n{out.stderr[-3000:]}")
    if REFIT or not fit_path.is_file():
        result = fit_latents(latent_path, spec, model, sub)
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(fit_path, index=False)
    return pd.read_csv(fit_path).iloc[0].to_dict()


def main():
    if MODEL not in (*SPECS, "both"):
        raise ValueError('MODEL must be "alpha_0", "alpha_1", or "both"')
    if N_SOBOL < 2 or N_SOBOL & (N_SOBOL - 1):
        raise ValueError("N_SOBOL must be a power of two")
    if GAMMA_SIGN not in (-1, 1):
        raise ValueError("GAMMA_SIGN must be -1 or +1")
    subjects, n_choices = load_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    helper_path = OUTPUT_ROOT / "_extract_alpha_latents.jl"
    helper_path.write_text(JULIA_HELPER, encoding="utf-8")

    config = {
        "batch": BATCH, "model": MODEL, "sobol": N_SOBOL,
        "starts": N_LOCAL_STARTS, "particles": N_PARTICLES,
        "samples": N_SAMPLES, "seed": SEED, "gamma_sign": GAMMA_SIGN,
        "trial_sha256": sha256(TRIAL_CSV), "prepared_sha256": sha256(MODEL_TRIAL_CSV),
        "source_julia_sha256": sha256(SOURCE_JULIA),
        "particle_filter_sha256": sha256(SOURCE_JULIA.with_name("particle_filter_child.jl")),
        "helper_sha256": hashlib.sha256(JULIA_HELPER.encode()).hexdigest(),
    }
    # Protect against silently reusing latents or old fits after configuration changes.
    config_path = OUTPUT_ROOT / "fit_config.json"
    if not REFIT and config_path.is_file():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        old.pop("model", None)
        new = config.copy()
        new.pop("model", None)
        if old != new:
            raise RuntimeError("Settings or input files changed; set REFIT=True to regenerate fits")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    selected = list(SPECS) if MODEL == "both" else [MODEL]
    all_results = []
    for model in selected:
        spec = SPECS[model]
        print(f"Fitting {model}: alpha={spec['alpha']}, free={spec['free']}; {len(subjects)} subjects")
        finished, failures = [], []
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            tasks = {pool.submit(run_subject, model, spec, sub, helper_path): sub for sub in subjects}
            for future in as_completed(tasks):
                sub = tasks[future]
                try:
                    row = future.result()
                    finished.append(row)
                    print(f"  sub {sub}: NLL={float(row['neg_llh']):.3f}")
                except Exception as e:
                    failures.append({"sub_num": sub, "error": str(e)})
                    print(f"  sub {sub}: ERROR: {e}")
        folder = OUTPUT_ROOT / model
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["sub_num", "error"]).to_csv(folder / "failures.csv", index=False)
        if finished:
            result = pd.DataFrame(finished).sort_values("sub_num")
            result.to_csv(folder / "summary.csv", index=False)
            all_results.append(result)
            print(f"Saved {len(result)}/{len(subjects)}: {folder / 'summary.csv'}")
        if failures:
            print(f"{len(failures)} failed. See {folder / 'failures.csv'}")

    if not all_results:
        raise RuntimeError("No successful fits")
    # Include previously completed model summaries when running one model at a time.
    # Their settings are protected by fit_config.json above.
    combined = []
    for name in SPECS:
        summary = OUTPUT_ROOT / name / "summary.csv"
        if summary.is_file():
            combined.append(pd.read_csv(summary))
    result = pd.concat(combined, ignore_index=True)
    result.to_csv(OUTPUT_ROOT / "all_model_fits.csv", index=False)

    if QUESTIONNAIRE_CSV.is_file():
        q = pd.read_csv(QUESTIONNAIRE_CSV, dtype={"fileID": str})
        if {"sub_num", "fileID"}.issubset(q.columns) and q.sub_num.is_unique and q.fileID.is_unique:
            result = result.merge(q[["sub_num", "fileID"]], on="sub_num", how="left", validate="many_to_one")
            if result.fileID.isna().any():
                print("WARNING: Some sub_num values are missing from questionnaire; no questionnaire copy was created")
            else:
                result.to_csv(OUTPUT_ROOT / "all_model_fits.csv", index=False)
                fields = ["gamma_base", "gamma_coef", "beta", "epsilon", "neg_llh", "AIC", "BIC"]
                wide = q.copy()
                for name in SPECS:
                    x = result[result.model == name][["sub_num"] + fields].copy()
                    if not x.empty:
                        x = x.rename(columns={v: f"{name}_{v}" for v in fields})
                        wide = wide.merge(x, on="sub_num", how="left", validate="one_to_one")
                wide.to_csv(OUTPUT_ROOT / "replication_questionnaires_with_alpha_models.csv", index=False)
                print("Saved questionnaire copy with model-specific parameters (original unchanged)")
        else:
            print("Questionnaire ID mapping not available or not unique; results saved by sub_num only")

    # Compare only subjects with successful fits in BOTH models.
    if set(result.model.unique()) == set(SPECS):
        a = result[result.model == "alpha_0"]
        b = result[result.model == "alpha_1"]
        c = a.merge(b, on="sub_num", how="inner", suffixes=("_alpha0", "_alpha1"))
        if len(c):
            if not (c.n_choices_alpha0 == c.n_choices_alpha1).all():
                raise ValueError("Cannot compare models with different numbers of choices")
            c["delta_AIC_alpha1_minus_alpha0"] = c.AIC_alpha1 - c.AIC_alpha0
            c["delta_BIC_alpha1_minus_alpha0"] = c.BIC_alpha1 - c.BIC_alpha0
            c["preferred_by_AIC"] = np.where(c.delta_AIC_alpha1_minus_alpha0 < 0,
                                              "alpha_1", "alpha_0")
            c.to_csv(OUTPUT_ROOT / "model_comparison.csv", index=False)
            print(f"Comparison saved: {OUTPUT_ROOT / 'model_comparison.csv'}")
    print(f"Done. Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
