Installation (no overwrites of original child archive):
Copy fit_alpha_models.py, replication_choice_fit_child.jl and particle_filter_child.jl into model_fits/ beside the existing child particle_filter.jl.
Input: model_fits/data/new_pilot_data_child.csv
Output: model_fits/alpha_model_results/
Choose MODEL = "both", "alpha_0", or "alpha_1" at the top of fit_alpha_models.py.
run in notebook: %run "fit_alpha_models.py" (from model_fits folder).

IMPORTANT differences from archived fits:
- alpha_0: CRP alpha=0, gamma_coef=0, free gamma_base and epsilon
- alpha_1: CRP alpha=0.2, gamma_coef free, free gamma_base and epsilon
- Both beta=1 and child timing uses 4.5 sec stay, 15.5 sec travel
- Uses inferred choices (except block-final censored), not recorded keypresses
- Uses one seeded CRP realization for likelihood prediction, not archived log-mean-exp across 1000 CRP realizations; estimates/AIC are not equivalent to archived historical fits.
- GAMMA_SIGN=+1 matches archived child code.
- The separate new_pilot_avg_child.csv is not needed for trial-level likelihood; the PRT data are contained in trial CSV.

Python: pip install numpy pandas scipy
Julia: julia -e 'using Pkg; Pkg.add(["CSV","DataFrames","Distributions","StatsBase","Sobol"])'
