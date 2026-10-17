# CRP + adaptive discounting + softmax/lapse, fitted to observed choices.
# Inputs: <sub_num> <trial_csv> <output_csv> <sobol_points> <particles> <samples> <seed>
# Standalone fitting script. Uses the particle-filter primitives provided in model.zip.
# Julia packages: CSV, DataFrames, Distributions, StatsBase, Sobol.

using CSV, DataFrames, Distributions, Statistics, StatsBase, Random, Sobol
include(joinpath(@__DIR__, "particle_filter_child.jl"))

const BOUNDS = [
    (0.0, 10.0),   # alpha (positive CRP concentration)
    (-10.0, 10.0),  # gamma_base
    (-3.0, 3.0),    # gamma_coef: allow either sign
    #(1, 1.0000000001),    # beta: inverse temperature
    (0.0, 0.10),    # epsilon: lapse probability
]

# Match the supplied model_with_beta_and_epsilon.jl, including its minus sign.
logistic(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
function normalize_logweights(v)
    finite = isfinite.(v)
    if !any(finite)
        return fill(inv(length(v)), length(v))
    end
    m = maximum(v[finite])
    x = [isfinite(y) ? exp(y - m) : 0.0 for y in v]
    s = sum(x)
    return (s > 0 && isfinite(s)) ? x ./ s : fill(inv(length(v)), length(v))
end

# The original posterior is prior(cluster) * likelihood(decays | cluster).
function cluster_posteriors(particles, N, alpha, decays)
    out = Vector{Vector{Float64}}(undef, length(particles))
    for (j, p) in enumerate(particles)
        priors = log_prior_particle(p, N, alpha)
        likes = isempty(decays) ? zeros(length(priors)) : log_likelihood_particle(p, decays)
        out[j] = normalize_logweights(priors .+ likes)
    end
    return out
end

# Entropy of the multinomial cluster-count distribution, as in the source model.
# Algebraically this is H = -log(n!) + n * H(p) + Σ_i E[log(X_i!)],
# where each marginal X_i ~ Binomial(n,p_i). Precompute the binomial terms
# for p_i = count_i/n once, instead of calling entropy(Multinomial(...)) in
# every trial/optimizer evaluation. This is the SAME quantity (not H(p)).
const MULTINOMIAL_ENTROPY_CACHE = Dict{Int, Tuple{Vector{Float64},Float64}}()
function sample_count_entropy(counts::Vector{Int})
    n = sum(counts)
    n > 0 || return 0.0
    expectations, log_n_factorial = get!(MULTINOMIAL_ENTROPY_CACHE, n) do
        log_facts = zeros(n + 1)
        for j in 1:n
            log_facts[j+1] = log_facts[j] + log(j)
        end
        table = zeros(n + 1)
        for c in 1:n
            b = Binomial(n, c/n)
            table[c+1] = sum(pdf(b, j)*log_facts[j+1] for j in 1:n)
        end
        (table, log_facts[end])
    end
    entropy_categories = 0.0
    for c in counts
        if c > 0
            p = c/n
            entropy_categories -= p*log(p)
        end
    end
    return max(0.0, -log_n_factorial + n*entropy_categories +
                    sum(expectations[c+1] for c in counts))
end

# Draw from the posterior predictive over next-reward decay.
# Keep the source model's uncertainty definition exactly (see function above).
function predictive_samples(particles, weights, posterior, n_samples)
    samples = Vector{Float64}(undef, n_samples)
    assignments = Vector{Int}(undef, n_samples)
    wp = Categorical(weights)
    for s in 1:n_samples
        p_idx = rand(wp)
        k = rand(Categorical(posterior[p_idx]))
        part = particles[p_idx]
        if k > length(part.cluster_mean)
            mu, var = part.hyper_mu, part.hyper_var
        else
            mu, var = part.cluster_mean[k], part.cluster_variance[k]
        end
        samples[s] = rand(Normal(mu, sqrt(max(var, 1e-12))))
        assignments[s] = k
    end
    counts = zeros(Int, maximum(assignments))
    for k in assignments
        counts[k] += 1
    end
    uncert = sample_count_entropy(counts)
    return mean(samples), uncert
end

# Particle-filter update after observing a planet's complete decay sequence.
# A safe variant of the source update: deepcopy independently, update sufficient
# statistics BEFORE appending the current planet, and keep resampled weights uniform.
function update_particles(particles, weights, posterior, decays)
    isempty(decays) && return particles, weights
    n = length(particles)
    replacements = Vector{Particle}(undef, n)
    anc_dist = Categorical(weights)
    for i in 1:n
        ancestor = rand(anc_dist)
        p = deepcopy(particles[ancestor])
        k = rand(Categorical(posterior[ancestor]))
        if k > length(p.cluster_mean)
            push!(p.cluster_mean, p.hyper_mu)
            push!(p.cluster_variance, p.hyper_var)
            push!(p.cluster_points, Float64[])
            push!(p.cluster_counts, 0)
            push!(p.tau, p.hyper_tau)
        end
        # Original Gaussian conjugate-update equations, with the existing
        # cluster's observations counted before the new planet is appended.
        old_n = length(p.cluster_points[k])
        n_new = length(decays)
        xbar = mean(decays)
        prior_mu, prior_var, prior_tau = p.cluster_mean[k], p.cluster_variance[k], p.tau[k]
        mu = (old_n * prior_mu + n_new * xbar) / (old_n + n_new)
        tau = prior_tau + n_new
        variance_numerator = (prior_tau * prior_var +
                              sum((x - xbar)^2 for x in decays) +
                              (old_n * n_new) / (old_n + n_new) * (prior_mu - xbar)^2)
        var = variance_numerator / tau
        p.cluster_mean[k] = mu
        p.cluster_variance[k] = max(var, 1e-12)
        p.tau[k] = tau
        append!(p.cluster_points[k], decays)
        p.cluster_counts[k] += 1
        replacements[i] = p
    end
    return replacements, fill(1.0 / n, n)
end

function prepare_subject(path, sub_num)
    all_data = CSV.read(path, DataFrame)
    required = [:sub_num, :block, :true_planet, :stay_num, :reward]
    missing_cols = setdiff(required, propertynames(all_data))
    isempty(missing_cols) || error("Missing required columns: $(missing_cols)")
    data = all_data[all_data.sub_num .== sub_num, :]
    nrow(data) > 0 || error("Subject $(sub_num) missing from trial file")
    any(c -> any(ismissing, data[!, c]), required) && error("Subject contains missing trial fields")
    data = sort(data, [:block, :true_planet, :stay_num])
    groups = groupby(data, [:block, :true_planet], sort=true)
    planets = []
    for g in groups
        stays = Int.(g.stay_num)
        stays == collect(0:(length(stays)-1)) ||
            error("Noncontiguous stay_num for sub $(sub_num), block $(g.block[1]), planet $(g.true_planet[1])")
        if :galaxy in propertynames(g)
            length(unique(g.galaxy)) == 1 || error("Galaxy changes within planet")
        end
        choices = if :observed_choice in propertynames(g)
            Union{Missing,Int}[ismissing(v) ? missing : Int(v) for v in g.observed_choice]
        else
            # Fallback (only if no explicit choices supplied by preprocessing).
            Union{Missing,Int}[fill(1, length(g)-1)..., missing]
        end
        any(x -> !ismissing(x) && !(x in (0,1)), choices) && error("Invalid choice value")
        push!(planets, (block=Int(g.block[1]), rewards=Float64.(g.reward), choices=choices))
    end
    length(planets) >= 2 || error("Too few observed planets")
    if !(:observed_choice in propertynames(data))
        # Infer a leave after the final dig if a next planet was observed.
        for i in eachindex(planets)
            if i < length(planets) && planets[i+1].block == planets[i].block
                planets[i].choices[end] = 0
            end
        end
    end
    n_choices = sum(count(!ismissing, p.choices) for p in planets)
    n_choices > 0 || error("No observed choices")
    return planets, nrow(data), n_choices
end

