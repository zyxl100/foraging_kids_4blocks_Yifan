
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
