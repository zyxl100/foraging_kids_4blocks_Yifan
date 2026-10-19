"""FULL closed-loop child-foraging agent simulation + alpha-model recovery.

Place next to fit_alpha_models.py, replication_choice_fit_child.jl,
and particle_filter_child.jl; input at data/new_pilot_data_child.csv.

Run: python run_full_agent_recovery.py
Needs: Python numpy pandas scipy, Julia CSV DataFrames Distributions StatsBase Sobol.
See README_full_agent_recovery.md for important methodological limitations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import math
import subprocess
import traceback

import numpy as np
import pandas as pd

import fit_alpha_models as fm

# =========================== USER SETTINGS ============================
GENERATING_MODELS = ("alpha_0", "alpha_1")  # Either model, or both
N_AGENTS_PER_MODEL = 29                 # PILOT: 5, FINAL: e.g. 50-100

# TRUE parameter ranges; narrower than bounds if desired, NEVER wider.
TRUE_PARAMETER_RANGES = {
    "gamma_base": (-4.0, 4.0),
    "gamma_coef": (-1.5, 1.5),          # Used only for alpha_1
    "epsilon": (0.0, 0.10),
}

# Generation: CRP + environment + agents' policy
N_PARTICLES_GENERATE = 200
N_SAMPLES_GENERATE = 20

# Fitting: independent CRP realization on EACH simulated trajectory
N_PARTICLES_FIT = 200
N_SAMPLES_FIT = 200
N_SOBOL = 128                   # Power of 2
N_LOCAL_STARTS = 10
WORKERS = 2                          # Julia is memory-intensive; increase cautiously

# Four-block task: source archive defaults
BLOCK_SECONDS = 360.0
MAX_PLANETS_PER_BLOCK = 20
MAX_HARVESTS_PER_PLANET = 100          # Abort an infinite/implausible trajectory

# Reproducibility and output (change RUN_NAME for a different experiment)
SIMULATION_SEED = 20261018
FIT_SEED = 84017                      # DIFFERENT from simulation seed
RUN_NAME = "pilot_v1"
OUT_DIR = fm.BASE / "full_agent_recovery" / RUN_NAME
DATA_CSV = fm.BASE / "data" / "new_pilot_data_child.csv"
JULIA = fm.JULIA                     # Or full path to julia.exe
# =====================================================================

# Original child-model definitions: alpha_0=(alpha=0, gamma_coef=0),
# alpha_1=(alpha=0.2, free gamma_coef). Beta is exactly 1 for BOTH.
FITTING_MODELS = ("alpha_0", "alpha_1")

# This is a stand-alone Julia *agent simulator*. It includes the EXACT same
# child PF and predictive/uncertainty definitions as the empirical fitter.
# Agent decisions cause new planet arrivals; no empirical reward trajectories
# are replayed. Galaxy block probabilities are supplied by Python.
JULIA_SIMULATOR = r'''
using CSV, DataFrames, Distributions, Random, Statistics
include(ARGS[1])

function main()
    out_csv = ARGS[2]
    agent_id = parse(Int, ARGS[3])
    alpha = parse(Float64, ARGS[4])
    g_base = parse(Float64, ARGS[5])
    g_coef = parse(Float64, ARGS[6])
    epsilon = parse(Float64, ARGS[7])
    num_particles = parse(Int, ARGS[8])
    n_samples = parse(Int, ARGS[9])
    seed = parse(Int, ARGS[10])
    env_csv = ARGS[11]
    block_seconds = parse(Float64, ARGS[12])
    max_planets = parse(Int, ARGS[13])
    max_harvests = parse(Int, ARGS[14])

    Random.seed!(seed)
    env = sort(CSV.read(env_csv, DataFrame), :block)
    blocks = Int.(env.block)
    blocks == collect(1:length(blocks)) || error("Environment blocks must be consecutive 1..B")
    0 <= epsilon <= 1 || error("Invalid epsilon")

    # Source child task: Timing(harvest=3, travel=10, iti=1.5, alien=5.5).
    harvest_cost = 4.5
    travel_cost = 15.5
    alien_time = 5.5

    particles, weights = init_particle_filter(num_particles, 0.5, 0.25, 1.0, alpha)
    total_reward, total_time, N = 0.0, alien_time, 0

    output = DataFrame(
        sub_num=Int[], block=Int[], true_planet=Int[], planet=Int[],
        planet_in_block=Int[], galaxy=Int[], stay_num=Int[],
        reward=Float64[], simulated_choice=Int[],
        observed_choice=Union{Missing,Int}[],
        v_stay_sim=Float64[], leave_scale_sim=Float64[], U_sim=Float64[]
    )

    for b in blocks
        er = env[env.block .== b, :]
        nrow(er) == 1 || error("Expected one row per block")
        prob = Float64[er.p_poor[1], er.p_neutral[1], er.p_rich[1]]
        isapprox(sum(prob), 1.0; atol=1e-8) || error("Galaxy probabilities do not sum to one")
        galaxy_dist = Categorical(prob)
        elapsed = alien_time
        visited = 0
        block_last_row = 0

        # An agent can reach different numbers/types of planets depending on choices.
        while elapsed < block_seconds && visited < max_planets
            galaxy = rand(galaxy_dist) - 1           # 0=poor, 1=neutral, 2=rich
            reward = Float64(round(clamp(rand(Normal(100, 5)), 0, 135)))
            decays = Float64[]
            last_post = nothing
            stay_num = 0

            while true
                # Reward is received before the subsequent stay/leave choice.
                total_reward += reward
                total_time += harvest_cost
                elapsed += harvest_cost
                if stay_num > 0
                    # The decay was appended when the reward was generated.
                    @assert !isempty(decays)
                end
                post = cluster_posteriors(particles, N, alpha, decays)
                last_post = post
                predicted_decay, U = predictive_samples(particles, weights, post, n_samples)
                v_stay = predicted_decay * reward
                leave_scale = (total_reward / total_time) * harvest_cost
                gamma = logistic(g_base + g_coef * U)  # + sign: archived child model
                p_stay = (1-epsilon)*logistic(v_stay-leave_scale*gamma) + epsilon/2
                choice = rand() < clamp(p_stay, 1e-12, 1-1e-12) ? 1 : 0

                push!(output, (
                    agent_id, b, N, N, visited, galaxy, stay_num, reward,
                    choice, choice, v_stay, leave_scale, U
                ))
                block_last_row = nrow(output)
                if choice == 0
                    break
                end
                stay_num += 1
                stay_num < max_harvests || error("Too many harvests on one planet; parameter region may be degenerate")
                decay = if galaxy == 0
                    rand(Beta(13, 51))
                elseif galaxy == 1
                    rand(Beta(50, 50))
                else
                    rand(Beta(50, 12))
                end
                new_reward = Float64(round(reward * decay))
                push!(decays, reward == 0.0 ? 1.0 : new_reward/reward)
                reward = new_reward
            end

            if !isempty(decays)
                particles, weights = update_particles(particles, weights, last_post, decays)
            end
            N += 1
            visited += 1
            total_time += travel_cost
            elapsed += travel_cost
        end
        block_last_row > 0 || error("A block had no planets")
        # The observed child data cannot identify the final block decision;
        # mimic that censoring exactly in the fitting data.
        output.observed_choice[block_last_row] = missing
    end

    mkpath(dirname(out_csv))
    tmp = out_csv * ".tmp"
    CSV.write(tmp, output)
    mv(tmp, out_csv; force=true)
    println("SIMULATED agent=$agent_id alpha=$alpha rewards=$(nrow(output)) " *
            "choices=$(count(!ismissing, output.observed_choice)) planets=$N")
end
main()
'''


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_settings() -> None:
    if not GENERATING_MODELS or any(m not in fm.SPECS for m in GENERATING_MODELS):
        raise ValueError(f"Unknown GENERATING_MODELS: {GENERATING_MODELS}")
    if N_AGENTS_PER_MODEL < 1 or N_SOBOL < 2 or N_SOBOL & (N_SOBOL - 1):
        raise ValueError("N_AGENTS_PER_MODEL >= 1; N_SOBOL must be a power of two")
    if not (1 <= N_LOCAL_STARTS <= N_SOBOL):
        raise ValueError("N_LOCAL_STARTS must lie in 1..N_SOBOL")
    if min(N_PARTICLES_GENERATE, N_PARTICLES_FIT, N_SAMPLES_GENERATE, N_SAMPLES_FIT, WORKERS) < 1:
        raise ValueError("Particles, samples and workers must be positive")
    if set(TRUE_PARAMETER_RANGES) != {"gamma_base", "gamma_coef", "epsilon"}:
        raise ValueError("TRUE_PARAMETER_RANGES must define gamma_base, gamma_coef, epsilon")
    if BLOCK_SECONDS <= 0 or MAX_PLANETS_PER_BLOCK < 1 or MAX_HARVESTS_PER_PLANET < 2:
        raise ValueError("Invalid environment limits")
    for p, lims in TRUE_PARAMETER_RANGES.items():
        if p not in fm.BOUNDS:
            raise ValueError(f"Unknown free parameter: {p}")
        lo, hi = lims
        glo, ghi = fm.BOUNDS[p]
        if not (glo <= lo < hi <= ghi):
            raise ValueError(f"True {p} range {lims} must lie inside fitter bounds {(glo, ghi)}")
    if not DATA_CSV.is_file():
        raise FileNotFoundError(f"Missing child trial data: {DATA_CSV}")
    for f in (fm.SOURCE_JULIA, fm.SOURCE_JULIA.with_name("particle_filter_child.jl")):
        if not f.is_file():
            raise FileNotFoundError(f"Missing Julia model file: {f}")
    if fm.GAMMA_SIGN != 1:
        raise ValueError("This agent simulator implements GAMMA_SIGN=+1. Set the existing fitter to +1.")
    if fm.SPECS["alpha_0"]["alpha"] != 0.0 or fm.SPECS["alpha_1"]["alpha"] != 0.2:
        raise ValueError("Expected original child alpha-model definitions (0.0, 0.2)")
    if fm.SPECS["alpha_0"]["free"] != ["gamma_base", "epsilon"] or fm.SPECS["alpha_1"]["free"] != ["gamma_base", "gamma_coef", "epsilon"]:
        raise ValueError("The fitter's free-parameter specifications changed")


def environment_probs() -> pd.DataFrame:
    """Estimate per-block galaxy frequencies from DISTINCT empirical planets."""
    d = pd.read_csv(DATA_CSV)
    required = {"sub_num", "block", "true_planet", "galaxy"}
    if not required.issubset(d.columns):
        raise ValueError(f"Missing trial columns: {required - set(d.columns)}")
    planets = d.drop_duplicates(["sub_num", "block", "true_planet"])
    if planets.galaxy.isna().any() or not set(planets.galaxy.astype(int)).issubset({0, 1, 2}):
        raise ValueError("Invalid galaxy categories; expected 0,1,2")
    counts = pd.crosstab(planets.block, planets.galaxy).reindex(columns=[0, 1, 2], fill_value=0)
    blocks = list(range(1, len(counts)+1))
    if sorted(int(v) for v in counts.index) != blocks:
        raise ValueError("Expected child task blocks numbered 1..B")
    probs = counts.div(counts.sum(axis=1), axis=0)
    probs.columns = ["p_poor", "p_neutral", "p_rich"]
    probs.index.name = "block"
    return probs.reset_index()


def true_parameters(model: str, agent_index: int) -> dict:
    model_id = 0 if model == "alpha_0" else 1
    rng = np.random.default_rng(np.random.SeedSequence([SIMULATION_SEED, model_id, agent_index]))
    vals = {k: float(rng.uniform(*lims)) for k, lims in TRUE_PARAMETER_RANGES.items()}
    vals["gamma_coef"] = vals["gamma_coef"] if model == "alpha_1" else 0.0
    return vals


def run_command(args: list[str], name: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {result.returncode}):\n{result.stderr[-5000:]}\n{result.stdout[-2000:]}")


def job_path(model: str, index: int) -> Path:
    return OUT_DIR / "runs" / f"gen_{model}_agent_{index:04d}"


def run_one(model: str, index: int, simulator_path: Path, helper_path: Path, env_path: Path) -> dict:
    """One independently simulated *trajectory*, re-evaluated under BOTH models."""
    folder = job_path(model, index)
    result_path = folder / "fits.csv"
    if result_path.is_file():
        result = pd.read_csv(result_path)
        if len(result) == 2 and set(result.fitting_model) == set(FITTING_MODELS):
            return {"model": model, "agent": index, "status": "cached"}
        raise ValueError(f"Incomplete fit file: {result_path}")

    folder.mkdir(parents=True, exist_ok=True)
    agent_id = (0 if model == "alpha_0" else 1000000) + index
    params = true_parameters(model, index)
    source = fm.SOURCE_JULIA
    gen_seed = SIMULATION_SEED + agent_id * 13 + 101
    agent_csv = folder / "simulated_trials.csv"
    if not agent_csv.is_file():
        cmd = [str(JULIA), "--startup-file=no", str(simulator_path), str(source),
               str(agent_csv), str(agent_id), str(fm.SPECS[model]["alpha"]),
               str(params["gamma_base"]), str(params["gamma_coef"]), str(params["epsilon"]),
               str(N_PARTICLES_GENERATE), str(N_SAMPLES_GENERATE), str(gen_seed),
               str(env_path), str(BLOCK_SECONDS), str(MAX_PLANETS_PER_BLOCK),
               str(MAX_HARVESTS_PER_PLANET)]
        run_command(cmd, f"generate {model} agent {index}")
    trajectory = pd.read_csv(agent_csv)
    if len(trajectory) == 0 or trajectory.sub_num.nunique() != 1 or trajectory.block.nunique() != len(pd.read_csv(env_path)):
        raise ValueError(f"Invalid simulated trajectory: {agent_csv}")
    if (trajectory.groupby("block").observed_choice.apply(lambda s: s.isna().sum()) != 1).any():
        raise ValueError("Expected exactly one censored final decision per block")
    # Check all nonfinal planet choices: 1=stay inside, 0=leave when moving.
    s = trajectory.sort_values(["block", "true_planet", "stay_num"])
    for (_, _), g in s.groupby(["block", "true_planet"], sort=False):
        if g.stay_num.tolist() != list(range(len(g))):
            raise ValueError("Noncontiguous within-planet stay numbers")
        if (g.observed_choice.iloc[:-1] != 1).any():
            raise ValueError("Early leave before planet termination")
    n_choices = int(trajectory.observed_choice.notna().sum())
    fits = []
    for candidate in FITTING_MODELS:
        spec = fm.SPECS[candidate]
        latents = folder / f"latents_{candidate}.csv"
        if not latents.is_file():
            cmd = [str(JULIA), "--startup-file=no", str(helper_path), str(source),
                   str(agent_id), str(agent_csv), str(latents), str(spec["alpha"]),
                   str(N_PARTICLES_FIT), str(N_SAMPLES_FIT), str(FIT_SEED)]
            run_command(cmd, f"extract {candidate} for {model} agent {index}")
        fit = fm.fit_latents(latents, spec, candidate, agent_id)
        if int(fit["n_choices"]) != n_choices:
            raise ValueError(f"Choice count mismatch: {fit['n_choices']} != {n_choices}")
        fits.append({
            "generating_model": model, "fitting_model": candidate,
            "agent_index": index, "agent_id": agent_id,
            "true_alpha": fm.SPECS[model]["alpha"],
            "true_gamma_base": params["gamma_base"],
            "true_gamma_coef": params["gamma_coef"],
            "true_epsilon": params["epsilon"], "true_beta": 1.0,
            "fitted_alpha": fit["alpha"],
            "fitted_gamma_base": fit["gamma_base"],
            "fitted_gamma_coef": fit["gamma_coef"],
            "fitted_epsilon": fit["epsilon"], "fitted_beta": fit["beta"],
            "neg_llh": fit["neg_llh"], "AIC": fit["AIC"], "BIC": fit["BIC"],
            "n_choices": n_choices, "n_rewards": len(trajectory),
            "n_planets": int(trajectory.true_planet.nunique()),
            "n_blocks": int(trajectory.block.nunique()),
            "optimizer_converged": fit["optimizer_converged"],
        })
    tmp = result_path.with_suffix(".csv.tmp")
    pd.DataFrame(fits).to_csv(tmp, index=False)
    tmp.replace(result_path)
    return {"model": model, "agent": index, "status": "fitted", "n_rewards": len(trajectory)}


def summarize(jobs: list[tuple[str, int]]) -> None:
    available = [job_path(m, i) / "fits.csv" for m, i in jobs if (job_path(m, i) / "fits.csv").exists()]
    if not available:
        raise RuntimeError("No successful fits; inspect failures.csv")
    x = pd.concat([pd.read_csv(f) for f in available], ignore_index=True)
    x.to_csv(OUT_DIR / "recovery_all_fits.csv", index=False)
    same = x[x.generating_model == x.fitting_model].copy()
    for p in ("gamma_base", "gamma_coef", "epsilon"):
        same[f"error_{p}"] = same[f"fitted_{p}"] - same[f"true_{p}"]
    same.to_csv(OUT_DIR / "recovery_matched.csv", index=False)
    keys = ["generating_model", "agent_index"]
    left = x[x.fitting_model == "alpha_0"]
    right = x[x.fitting_model == "alpha_1"]
    comp = left.merge(right, on=keys, suffixes=("_alpha0", "_alpha1"), validate="one_to_one")
    if not (comp.n_choices_alpha0 == comp.n_choices_alpha1).all():
        raise ValueError("Both models must use the same choice counts")
    for c in ("AIC", "BIC"):
        delta = comp[f"{c}_alpha1"] - comp[f"{c}_alpha0"]
        comp[f"delta_{c}_alpha1_minus_alpha0"] = delta
        comp[f"selected_by_{c}"] = np.where(delta < 0, "alpha_1", "alpha_0")
        comp[f"correct_by_{c}"] = comp[f"selected_by_{c}"] == comp.generating_model
    comp.to_csv(OUT_DIR / "model_recovery.csv", index=False)
    print("\nCompleted simulated agents:", len(same), "/", len(jobs))
    print("Model recovery:")
    for c in ("AIC", "BIC"):
        print(f"  {c}: {comp[f'correct_by_{c}'].mean():.1%}")
        print(pd.crosstab(comp.generating_model, comp[f"selected_by_{c}"]).to_string())
    print("Results:", OUT_DIR)


def main():
    validate_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = environment_probs()
    env_path = OUT_DIR / "environment_probabilities.csv"
    # Config fingerprints protect cached fits without deleting previous runs.
    config = {
        "generating_models": list(GENERATING_MODELS),
        "true_parameter_ranges": {k: list(v) for k,v in TRUE_PARAMETER_RANGES.items()},
        "n_particles_generate": N_PARTICLES_GENERATE,
        "n_samples_generate": N_SAMPLES_GENERATE,
        "n_particles_fit": N_PARTICLES_FIT,
        "n_samples_fit": N_SAMPLES_FIT,
        "n_sobol": N_SOBOL, "n_local_starts": N_LOCAL_STARTS,
        "simulation_seed": SIMULATION_SEED, "fit_seed": FIT_SEED,
        "block_seconds": BLOCK_SECONDS, "max_planets_per_block": MAX_PLANETS_PER_BLOCK,
        "max_harvests_per_planet": MAX_HARVESTS_PER_PLANET,
        "gamma_sign": fm.GAMMA_SIGN,
        "model_specs": fm.SPECS, "fit_bounds": {k: list(v) for k, v in fm.BOUNDS.items()},
        "task_source": "empirical per-block galaxy probabilities, ORIGINAL child reward generator and timing",
        "environment": env.to_dict(orient="records"),
        "data_hash": file_hash(DATA_CSV),
        "fitter_python_hash": file_hash(Path(fm.__file__)),
        "fitter_julia_hash": file_hash(fm.SOURCE_JULIA),
        "particle_filter_hash": file_hash(fm.SOURCE_JULIA.with_name("particle_filter_child.jl")),
        "simulator_hash": hashlib.sha256(JULIA_SIMULATOR.encode()).hexdigest(),
        "latent_helper_hash": hashlib.sha256(fm.JULIA_HELPER.encode()).hexdigest(),
    }
    config_path = OUT_DIR / "recovery_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Recovery settings/source changed: set a NEW RUN_NAME to protect earlier fits")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    env.to_csv(env_path, index=False)
    script_path = OUT_DIR / "_simulate_agent.jl"
    helper_path = OUT_DIR / "_extract_alpha_latents.jl"
    script_path.write_text(JULIA_SIMULATOR, encoding="utf-8")
    helper_path.write_text(fm.JULIA_HELPER, encoding="utf-8")

    # Reuse the EXACT Python likelihood and optimization code for both models.
    fm.N_SOBOL = N_SOBOL
    fm.N_LOCAL_STARTS = N_LOCAL_STARTS
    fm.N_PARTICLES = N_PARTICLES_FIT
    fm.N_SAMPLES = N_SAMPLES_FIT
    fm.SEED = FIT_SEED

    jobs = [(m, i) for m in GENERATING_MODELS for i in range(1, N_AGENTS_PER_MODEL+1)]
    print("Environment galaxy probabilities by block:\n", env.to_string(index=False), flush=True)
    print(f"Closed-loop agents: {len(jobs)} | fit models/agent: 2 | pending: "
          f"{sum(not (job_path(m,i)/'fits.csv').exists() for m,i in jobs)}", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, model, i, script_path, helper_path, env_path): (model, i)
                   for model, i in jobs}
        for future in as_completed(futures):
            model, i = futures[future]
            try:
                outcome = future.result()
                print(f"  {model} agent {i}: {outcome['status']}", flush=True)
            except Exception as exc:
                failures.append({"model": model, "agent": i, "error": str(exc)})
                print(f"  {model} agent {i}: ERROR {exc}", flush=True)
    pd.DataFrame(failures, columns=["model", "agent", "error"]).to_csv(OUT_DIR / "failures.csv", index=False)
    summarize(jobs)
    if failures:
        print(f"WARNING: {len(failures)} failed; see {OUT_DIR/'failures.csv'}")


if __name__ == "__main__":
    main()
