# module SAC
# export train

include("../training/utils.jl")
include("cart-pole.jl")
using .Utils
using .CartPole
using Flux
using CUDA
using Statistics: mean
using Dates: now
using Distributions
using BSON: @save, @load

cfg = load_config()

function evaluate(policy_network, state, norm_distribution, c)
    policy_preds = policy_network(state)
    μ = view(policy_preds, 1:action_size, :)
    log_σ = clamp.(view(policy_preds, action_size+1:2*action_size, :), -20, 2)
    σ = exp.(log_σ)
    z = (rand.(norm_distribution) |> gpu) .* exp.(log_σ) .+ μ
    logp_π  = view(sum(-(z .- μ).^2 ./ (2 .* σ .^ 2) .- log_σ .- c, dims=1), 1, :)
    logp_π = logp_π .- view(sum(2 * (log(2f0) .- z .- log.(1 .+ exp.(-2 .* z))), dims=1), 1, :)
    return tanh.(z), logp_π
end

function calc_q_targets(policy, s′, r, t, q1_tgt, q2_tgt, γ, α, norm_distribution, c)
    a′, log_π = evaluate(policy, s′, norm_distribution, c)
    q′_input = vcat(s′, a′)
    q′ = min.(q1_tgt(q′_input), q2_tgt(q′_input))
    q′ = view(q′, 1, :)
    ŷ = r .+ γ .* (1 .- t) .* vec((q′ .- α .* log_π))
    return ŷ 
end

function get_checkpoint_idx(exp_name, model_n)

    # get the array of checkpoints
    checkpoints = readdir("models/$exp_name")

    if size(checkpoints)[1] > 0
        # get the highest index of the checkpoints
        name = "model$model_n"
        r = Regex("$name")
        idx = -1
        for i = eachindex(checkpoints)
            m = match(r, checkpoints[i])
            if m !== nothing
                m = match(r"_\d+", checkpoints[i]).match
                n = parse(Int, match(r"\d+", m).match)
                if n > idx
                    idx = n
                end
            end
        end
        if idx < 0
            idx = nothing
        end
        return idx
    else
        return nothing
    end
end

function setup_networks(exp_name, model_n, lr)
    n1 = 256
    n2 = 512

    idx = get_checkpoint_idx(exp_name, model_n)
    #checkpoints = readdir("models/$exp_name", join=true)

    if idx != nothing
        println("Starting training from checkpoint $idx")
        @load "models/$(exp_name)/model$(model_n)_$(idx).bson" cpu_q1_network cpu_q2_network cpu_q1_tgt cpu_q2_tgt cpu_policy_network
        q1_network = cpu_q1_network |> gpu
        q2_network = cpu_q2_network |> gpu
        q1_tgt = cpu_q1_tgt |> gpu
        q2_tgt = cpu_q2_tgt |> gpu
        policy_network = cpu_policy_network |> gpu
    else

        # Q-function networks
        q1_network = Chain(
            Dense(state_size + action_size, n1, relu),
            Dense(n1, n2, relu),
            Dense(n2, n2, relu),
            #Dense(n2, n2, relu),
            Dense(n2, n2, relu),
            Dense(n2, n1, relu),
            Dense(n1, 1)
        ) |> gpu
        q2_network = Chain(
            Dense(state_size + action_size, n1, relu),
            Dense(n1, n2, relu),
            Dense(n2, n2, relu),
            #Dense(n2, n2, relu),
            Dense(n2, n2, relu),
            Dense(n2, n1, relu),
            Dense(n1, 1)
        ) |> gpu
        q1_tgt = Chain(
            Dense(state_size + action_size, n1, relu),
            Dense(n1, n2, relu),
            Dense(n2, n2, relu),
            #Dense(n2, n2, relu),
            Dense(n2, n2, relu),
            Dense(n2, n1, relu),
            Dense(n1, 1)
        ) |> gpu
        q2_tgt = Chain(
            Dense(state_size + action_size, n1, relu),
            Dense(n1, n2, relu),
            Dense(n2, n2, relu),
            #Dense(n2, n2, relu),
            Dense(n2, n2, relu),
            Dense(n2, n1, relu),
            Dense(n1, 1)
        ) |> gpu

        # initialse the target q function network weights the same as the original q functions
        for i = 1:length(q1_tgt)
            q1_tgt[i].W[:,:] = q1_network[i].W
            q1_tgt[i].b[:] = q1_network[i].b
            q2_tgt[i].W[:,:] = q2_network[i].W
            q2_tgt[i].b[:] = q2_network[i].b
        end

        # Policy network
        policy_network = Chain(
            Dense(state_size, n1, relu),
            Dense(n1, n2, relu),
            Dense(n2, n2, relu),
            #Dense(n2, n2, relu),
            Dense(n2, n2, relu),
            Dense(n2, n1, relu),
            Dense(n1, 2 * action_size)
        ) |> gpu
        
    end

    # Setup optimisers

    α_opt = ADAM(lr)
    policy_opt = ADAM(lr) |> gpu
    q1_opt = ADAM(lr) |> gpu
    q2_opt = ADAM(lr) |> gpu

    return q1_network, q2_network, q1_opt, q1_opt, q1_tgt, q2_tgt, policy_network, policy_opt, α_opt
end

function train(exp_name, model_n)

    # set hyper parameters
    τ = 0.995f0
    α = 0.2f0
    γ = 0.99f0
    batch_size = 100
    replay_size = 10^6
    lr=0.001
    start_steps=0 # 10000
    update_after=100
    checkpoint_every = 10000
    update_α = false
    reward_scale = 1

    # setup networks
    q1_network, q2_network, q1_opt, q2_opt, q1_tgt, q2_tgt, policy_network, policy_opt, α_opt = setup_networks(exp_name, model_n, lr)
    log_α = 0
    target_entropy = -action_size
    cpu_policy_network = policy_network |> cpu

    # set constants
    c::Float32 = log(sqrt(2 * π))
    max_steps = 200
    epsilon::Float32 = 0.000001

    # set up checkpoints folder
    try
        mkdir(string("models/", exp_name))
    catch
    end

    # initialise buffers
    N = replay_size
    n = 1
    states_t = Array{Float32}(undef, state_size, 0)
    states_t1 = Array{Float32}(undef, state_size, 0)
    actions_t = Array{Float32}(undef, action_size, 0)
    rewards_t = Array{Float32}(undef, 0)
    unfinished_t1 = Array{Float32}(undef, 0)
    state = Array{Float32}(undef, state_size)
    norm_state = Array{Float32}(undef, state_size)
    next_state = Array{Float32}(undef, state_size)
    init_states = Array{Float32}(undef, state_size, 0)
    prev_state = Array{Float32}(undef, state_size)
    pen_state = Array{Float32}(undef, state_size)

    # setup Normal distribution for selection of normalised actions in each batch
    μ = Array{Float32}(undef, action_size, batch_size)
    σ = Array{Float32}(undef, action_size, batch_size)
    for i = 1:batch_size
        for j = 1:action_size
            μ[j,i] = 0
            σ[j,i] = 1
        end
    end
    norm_distribution = Normal.(μ, σ)

    # train
    st = now()
    idx = 0
    episodes=0
    n_steps=0
    while true

        # choose an opponent
        opponent = choose_opponent(exp_name)

        # run an episode
        state = [0.0,0.0,0.0,0.0]
        reward = 0

        if episodes % 10 == 0
            println("Episodes $episodes | Average steps ", n_steps/10)
            n_steps=0
        end

        episodes += 1

        for j = 1:max_steps

            # get the normalised state
            norm_state[:] = state

            # add the initial state to the buffer
            states_t = cat(states_t, norm_state, dims=2)

            # select an action by sampling from the policy
            if n < start_steps
                action = rand(action_size) .* 2 .- 1
            else
                policy_pred = cpu_policy_network(norm_state)
                μ = policy_pred[1:action_size]
                σ = exp.(clamp.(policy_pred[action_size+1:2*action_size], -20, 2))
                d = Normal.(μ, σ)
                action = tanh.(rand.(d))
            end

            step_reward, state, done = env_step(state, action)
            reward += step_reward * reward_scale

            # add the new state, action, reward and done to the buffers
            next_state[:] = state
            states_t1 = cat(states_t1, next_state, dims=2)
            actions_t = cat(actions_t, action, dims=2)
            rewards_t = cat(rewards_t, step_reward * reward_scale, dims=1)
            if done
                unfinished_t1 = cat(unfinished_t1, 0, dims=1)
                n_steps += j
            else
                unfinished_t1 = cat(unfinished_t1, 1, dims=1)
            end

            # go to next step
            n += 1

            # update networks
            if (n > update_after) && (n > batch_size * 2)
                # choose a random batch from the buffer
                l = size(states_t)[2]
                idxs = rand(1:l, batch_size)
                s = states_t[:, idxs] |> gpu
                s′ = states_t1[:, idxs] |> gpu
                a = actions_t[:, idxs] |> gpu
                r = rewards_t[idxs] |> gpu
                t = unfinished_t1[idxs] |> gpu
                q_input = cat(s, a, dims=1)

                # calculate the targets for the q function
                ŷ = calc_q_targets(policy_network, s′,  r, t, q1_tgt, q2_tgt, γ, α, norm_distribution, c)

                # update the q functions
                q1_params = Flux.params(q1_network)
                q1_grad = gradient(Flux.params(q1_network)) do
                    q1 = q1_network(q_input)
                    Flux.mse(q1, ŷ)
                end
                Flux.update!(q1_opt, q1_params, q1_grad)
                q2_params = Flux.params(q2_network)
                q2_grad = gradient(Flux.params(q2_network)) do
                    q2 = q2_network(q_input)
                    Flux.mse(q2, ŷ)
                end
                Flux.update!(q2_opt, q2_params, q2_grad)

                # update the policy
                p_params = Flux.params(policy_network)
                p_grad = gradient(Flux.params(policy_network)) do
                    a, log_π = evaluate(policy_network, s, norm_distribution, c)
                    q_input = cat(s, a, dims=1)
                    q = min.(q1_network(q_input), q2_network(q_input))
                    q = view(q, 1, :)
                    mean(α .* log_π .- q)
                end
                Flux.update!(policy_opt, p_params, p_grad)

                # update α
                if update_α
                    ps_α = Flux.params(log_α)
                    grad_α = gradient(() -> mean(log_α .* (log_prob .- target_entropy)), ps_α)
                    Flux.update!(α_opt, ps_α, grad_α)
                    α = exp(log_α)
                end

                # copy back to the cpu
                cpu_policy_network = policy_network |> cpu

                # update the target network weights
                for (dest, src) in zip(
                    Flux.params([q1_tgt, q2_tgt]),
                    Flux.params([q1_network, q2_network]),
                )
                    dest .= τ .* dest .+ (1 - τ) .* src
                end

            end

            # shift buffer if full
            if size(states_t)[2] > 1000000 # TODO make these hyperparameters
                println("\n SHIFTING BUFFER \n")
                states_t = states_t[:,end-900000+1:end]
                states_t1 = states_t1[:,end-900000+1:end]
                actions_t = actions_t[:,end-900000+1:end]
                rewards_t = rewards_t[:,end-900000+1:end]
                unfinished_t1 = unfinished_t1[:,end-900000+1:end]
            end

            # periodically checkpoint the policy
            if n % checkpoint_every == 0
                # score_policy(cpu_policy_network, max_steps)
                cpu_q1_network = q1_network |> cpu
                cpu_q2_network = q2_network |> cpu
                cpu_q1_tgt = q1_tgt |> cpu
                cpu_q2_tgt = q2_tgt |> cpu
                idx += 1
                model_name = string("models/", exp_name , "/model", model_n, "_", idx, ".bson")
                println("Saving model: ", model_name, "                        ")
                @save model_name cpu_q1_network cpu_q2_network cpu_q1_tgt cpu_q2_tgt cpu_policy_network
            end

            # timing for performance tracking
            if false
            #if n > 100 && n % 100 == 0 & false
                print("n: $n | time for 100 updates: ", now() - st, "              \r")
                st = now()
            end 

            if done
                break
            end

        end
    end
end

exp_name = ARGS[1]
model_n = ARGS[2]

train(exp_name, model_n)
