
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
