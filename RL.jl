using ReinforcementLearning, ReinforcementLearningCore
using .ReinforcementLearningBase
using Statistics , Flux , Functors , StableRNGs ,  Random, Zygote, Distributions, LinearAlgebra, QuantumOptics, DifferentialEquations, DiffEqFlux, OrdinaryDiffEq, QuantumOpticsBase
using Flux: Chain, Dense
using IntervalSets: ClosedInterval 
using Plots
using LinearAlgebra
import QuantumOpticsBase: normalize

using PyCall
pushfirst!(PyVector(pyimport("sys")["path"]), "")
qw = pyimport("quantum_well")

# ────────────────────────────────────────────────────────────────────────────────
# RL ENVIRONMENT 
# ────────────────────────────────────────────────────────────────────────────────
mutable struct WellEnv <: AbstractEnv
    #lo stato visto da RL sono i parametri correnti [a, k, d, delta, ky]
    params::Vector{Float64} 
    reward::Float64
    done::Bool
    steps::Int
    max_steps::Int
    rng::StableRNG
end
 

function WellEnv(;max_steps=50)
    
    return WellEnv(copy(INITIAL_P), 0.0, false, 0, max_steps, StableRNG(123))
end


const PARAMS_MIN = [0.3,   50.0,   0.8, -5.0,  500.0]  # [a, k, ky, delta, d]
const PARAMS_MAX = [0.9, 3000.0,   1.2,  5.0, 3000.0]


const INITIAL_P  = [0.6, 1000.0,   1.0,  0.0, 1500.0]

RLBase.action_space(env::WellEnv) = Space([ClosedInterval(-1.0, 1.0) for _ in 1:5])


function RLBase.state(env::WellEnv)
    return @. 2.0 * (env.params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN) - 1.0
end



RLBase.is_terminated(env::WellEnv) = env.done || env.steps >= env.max_steps

function RLBase.reset!(env::WellEnv)

    env.params = copy(INITIAL_P) 
   
    env.steps = 0
    env.reward = 0.0
    env.done = false
    
    return RLBase.state(env)
end

# ────────────────────────────────────────────────────────────────────────────────
# PPO 
# ────────────────────────────────────────────────────────────────────────────────

struct Actor
    chain::Chain
end
Flux.@layer Actor



function Actor(state_dim::Int, action_dim::Int)
    chain = Chain(
        Dense(state_dim, 128, tanh),
        Dense(128, 64, tanh),
        Dense(64, 32, tanh),
        Dense(32, action_dim * 2)   # raw outputs: μ_raw, logσ_raw
    )
    Actor(chain)
end

function (actor::Actor)(state)
    x = actor.chain(state)
    action_dim = size(x, 1) ÷ 2

    if ndims(x) == 1
        μ_raw     = x[1:action_dim]
        logσ_raw  = x[action_dim+1:end]
    else
        μ_raw     = x[1:action_dim, :]
        logσ_raw  = x[action_dim+1:end, :]
    end

    # ====== stabilizzazione μ ======
    # μ_raw ∈ R → μ = tanh(μ_raw) ∈ [-1,1]
    μ = tanh.(μ_raw)

    # Se vuoi avere un controllo più fine:
    # μ_scale = 0.8
    # μ = μ_scale .* tanh.(μ_raw)

    # ====== stabilizzazione σ ======
    # limiti soft e piccoli: esplorazione moderata
    logσ_clipped = clamp.(logσ_raw, -8.0, -3.0)   # σ ∈ [0.0009, 0.36]
    σ            = exp.(logσ_clipped)

    return Normal.(μ, σ)
end

struct Critic
    chain::Chain
end
Flux.@layer Critic

function Critic(state_dim::Int)
    chain = Flux.Chain(
       Flux.Dense(state_dim, 128, relu),
        Flux.Dense(128, 64, relu),
       Flux.Dense(64, 32, relu),
        Flux.Dense(32, 1)
    )
    Critic(chain)
end

function (critic::Critic)(state)
    v = critic.chain(state)
    if ndims(v) > 1; return vec(v); else return first(v); end
end

atanh_clamped(x) = 0.5 * log((1 + clamp(x, -1 + 1e-6, 1 - 1e-6)) / (1 - clamp(x, -1 + 1e-6, 1 - 1e-6)))

function sample_squashed(dist)
    u = [rand(d) for d in dist] |> x -> reshape(x, size(dist))
    a = tanh.(u)
    log_base = logpdf.(dist, u)
    if ndims(log_base) == 1
        lp = sum(log_base) - sum(log.(1 .- a.^2 .+ 1e-6))
        return a, lp
    else
        lp = vec(sum(log_base, dims=1) .- sum(log.(1 .- a.^2 .+ 1e-6), dims=1))
        return a, lp
    end
end

function logprob_squashed(dist, a)
    u = atanh_clamped.(a)
    lp = sum(logpdf.(dist, u), dims=1) .- sum(log.(1 .- a.^2 .+ 1e-6), dims=1)
    return vec(lp)
end

function step!(env::WellEnv, action::AbstractVector)

    ## take the action from RL and build the params for the potential V
    # mini-variations in the step for the actions (?) instead of a 1 step fopr each episode

    step_size = 0.05
    param_ranges = PARAMS_MAX .- PARAMS_MIN
    delta_params = @. action * step_size * param_ranges

    new_params = env.params .+ delta_params
    env.params = clamp.(new_params, PARAMS_MIN, PARAMS_MAX)
    
    # da julia a python lo fa PyCall (?) vedi se funziona - pare di si 
    params_py = Dict(
            "a"     => env.params[1],
            "k"     => env.params[2],
            "ky"    =>  1500.0, #env.params[3],
            "delta" =>  0,      #env.params[4]
            "d"     => env.params[5]
        )




    V2D = qw.build_potential_matrix(x_grid, y_grid, params_py)

    single_res = qw.single_particle_eigenstates(T2D, V2D, Nstates)
    single_energies, single_vecs = single_res[1], single_res[2]

    LAM_AMBIG_TOL = 0.05  # states with |lambda-0.5| <= tol are tagged ambiguous

    loc_res = qw.localise_orbitals_projector_DVR(
            single_vecs, x_grid, y_grid,
            M=M_LOC, x_cut=X_CUT, smooth=SMOOTH_PL, sigma=SIGMA_PL
        )

    U_loc, vecs_loc, lam_loc, lr_labels = loc_res[1], loc_res[2], loc_res[3], loc_res[4]
    
    ambiguous_mu = findall(x -> abs(x - 0.5) <= LAM_AMBIG_TOL, lam_loc)
    n_ambiguous = length(ambiguous_mu)
    loc_quality = mean(@. 2.0 * abs(lam_loc - 0.5))


    mode_tags_spin = String[]
    for mu in 1:M_LOC
        push!(mode_tags_spin, lr_labels[mu]) # Up
        push!(mode_tags_spin, lr_labels[mu]) # Down
    end


    ## CI Interaction

    slater_basis = qw.build_slater_basis_sorted(Nstates, single_energies)
    n_compute    = min(N_CI_COMPUTE, length(slater_basis))
    H_slater     = qw.build_ci_hamiltonian(slater_basis, single_energies,
                                        single_vecs, K, n_compute)
    E2, C2 = eigh(H_slater)

    C2_spin_purified = qw.purify_degenerate_spin_subspaces(
    E2, C2, slater_basis[1:n_compute], energy_tol=1e-6, spin_tol=1e-8
    )

    n_state = 1
    U_spin = qw.spinorbital_U_from_spatial(U_loc) 
    U_spin_T = transpose(U_spin)


    Om_full = qw.ci_to_spinorbital_Omega(C2_spin_purified[:, n_state], slater_basis[1:n_compute], Nstates)
    Om_sub = qw.truncate_Omega_to_subspace(Om_full, M_LOC)


    Om_loc = U_spin_T * Om_sub * U_spin
    Om_loc = 0.5*(Om_loc - transpose(Om_loc))

   
    rho2_loc, pairs_spin = qw.Omega_to_rho2_pair(Om_loc)
    rhoL0, rhoL1, rhoL2, meta = qw.rhoL_from_rho2_pairs_spin(rho2_loc, pairs_spin, mode_tags_spin)


    
    
    ## Vincoli sui guess dei parametri dati da RL:

    #FPM email -> non contare stati in cui non abbiamo autovalori circa 0 o circa 1 ,
    # quindi se l autovalore è 0.5 gli stati non sono localizzati

    
    if loc_quality < 0.7 # Soglia arbitraria
        env.reward = -1.0 # Penalità per stati delocalizzati
    else
        E_acc, (p0,p1,p2) = qw.accessible_entropy(rhoL0, rhoL1, rhoL2)
        env.reward = E_acc
    end

    env.current_step += 1
    env.done = env.current_step >= 100

    return env.reward, env.done
end

function step_envs!(envs::Vector{QuantumEnv}, actions::Vector)
    N = length(envs)
    rewards = Vector{Float64}(undef, N)
    dones   = Vector{Bool}(undef, N)
    states  = Vector{Vector{Float64}}(undef, N)
    Threads.@threads for i in 1:N
        r, d = step!(envs[i], actions[i])
        rewards[i] = r
        dones[i]   = d
        states[i]  = RLBase.state(envs[i])
    end
    return states, rewards, dones
end

struct PPOPolicy
    actor::Actor
    critic::Critic
end
Functors.@functor PPOPolicy

mutable struct PPOBuffer
    states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
    rewards::Vector{Float64}
    dones::Vector{Bool}
    log_probs::Vector{Float64}
    values::Vector{Float64}
    env_ids::Vector{Int}
end

function PPOBuffer()
    PPOBuffer([], [], [], [], [], [], Int[])
end

function Base.empty!(buffer::PPOBuffer)
    empty!(buffer.states)
    empty!(buffer.actions)
    empty!(buffer.rewards)
    empty!(buffer.dones)
    empty!(buffer.log_probs)
    empty!(buffer.values)
    empty!(buffer.env_ids)
end

mutable struct PPOAgent
    policy::PPOPolicy
    actor_optimizer
    critic_optimizer
    actor_opt_state
    critic_opt_state
    gamma::Float64
    lambda::Float64
    clip_range::Float64
    entropy_loss_weight::Float64
    critic_loss_weight::Float64
    max_grad_norm::Float64
    n_rollout::Int
    n_env::Int
    n_update_epochs::Int
    mini_batch_size::Int
    rng::StableRNGs.StableRNG
    buffer::PPOBuffer
end
Functors.@functor PPOAgent (policy,)

function PPOAgent(
    actor::Actor,
    critic::Critic;
    actor_optimizer = Adam(1e-3),
    critic_optimizer = Adam(1e-3),
    gamma::Real = 0.99,
    lambda::Real = 0.95,
    clip_range::Real = 0.2,
    entropy_loss_weight::Real = 0.01,
    critic_loss_weight::Real = 0.5,
    max_grad_norm::Real = 0.5,
    n_rollout::Int = 2048,
    n_env::Int = 8,
    n_update_epochs::Int = 10,
    mini_batch_size::Int = 64,
    rng = StableRNG(123),
)
    policy = PPOPolicy(actor, critic)
    buffer = PPOBuffer()
    actor_opt_state  = Flux.setup(actor_optimizer,  policy.actor)
    critic_opt_state = Flux.setup(critic_optimizer, policy.critic)
    return PPOAgent(
        policy, actor_optimizer, critic_optimizer,
        actor_opt_state, critic_opt_state,
        float(gamma), float(lambda), float(clip_range),
        float(entropy_loss_weight), float(critic_loss_weight),
        float(max_grad_norm),
        n_rollout, n_env, n_update_epochs, mini_batch_size,
        rng, buffer
    )
end

function reset_opt_states!(agent::PPOAgent)
    agent.actor_opt_state  = Flux.setup(agent.actor_optimizer,  agent.policy.actor)
    agent.critic_opt_state = Flux.setup(agent.critic_optimizer, agent.policy.critic)
    return agent
end

# utility buffer
function _min_len(buf::PPOBuffer)
    minimum((
        length(buf.states), length(buf.actions), length(buf.rewards),
        length(buf.dones), length(buf.log_probs), length(buf.values),
        length(buf.env_ids)
    ))
end
function _trim_to!(buf::PPOBuffer, L::Int)
    buf.states     = buf.states[1:L]
    buf.actions    = buf.actions[1:L]
    buf.rewards    = buf.rewards[1:L]
    buf.dones      = buf.dones[1:L]
    buf.log_probs  = buf.log_probs[1:L]
    buf.values     = buf.values[1:L]
    buf.env_ids    = buf.env_ids[1:L]
end

function store_transition!(agent::PPOAgent, state, action, reward, done, logp, value; env_id::Int)
    push!(agent.buffer.states, state)
    push!(agent.buffer.actions, action)
    push!(agent.buffer.rewards, reward)
    push!(agent.buffer.dones, done)
    push!(agent.buffer.log_probs, logp)
    push!(agent.buffer.values, value)
    push!(agent.buffer.env_ids, env_id)
end

function store_step!(buffer::PPOBuffer, state, action, reward, done, log_prob, value)
    push!(buffer.states, state)
    push!(buffer.actions, action)
    push!(buffer.rewards, reward)
    push!(buffer.dones, done)
    push!(buffer.log_probs, log_prob)
    push!(buffer.values, value)
end

function select_action(agent::PPOAgent, state::Vector{Float64})
    dists = agent.policy.actor(state)
    μ = [d.μ for d in dists]
    σ = [d.σ for d in dists]

    a, logp = sample_squashed(dists)
    value = agent.policy.critic(state)

    return collect(a), (logp isa AbstractArray ? logp[1] : logp), value, μ, σ
end
function select_action_eval(agent::PPOAgent, state::Vector{Float64})
    # Ricava la distribuzione dell'attore
    dists = agent.policy.actor(state)
    
    # Prendi la media μ, NON campioniamo
    μ = [d.μ for d in dists]   # μ ∈ [-1, 1] grazie al tanh

    # Ritorna μ come azione deterministica
    return collect(μ)
end
function ready_to_update(agent::PPOAgent)
    return length(agent.buffer.rewards) >= agent.n_rollout
end

function update!(agent::PPOAgent; bootstrap_values_by_env::Vector{Float64})
    L = _min_len(agent.buffer)
    if L == 0
        empty!(agent.buffer); return
    end
    _trim_to!(agent.buffer, L)
    update_policy!(agent, bootstrap_values_by_env)
    empty!(agent.buffer)
end

function clear_buffer!(agent::PPOAgent)
    agent.buffer.states = []
    agent.buffer.actions = []
    agent.buffer.rewards = Float64[]
    agent.buffer.dones = Bool[]
    agent.buffer.values = Float64[]
    agent.buffer.log_probs = Float64[]
end

function compute_gae(rewards::Vector{Float64}, values::Vector{Float64}, dones::Vector{Bool}; γ=0.99, λ=0.95)
    T = length(rewards)
    @assert length(values) == T + 1 "Il vettore values deve avere lunghezza T+1"
    @assert length(dones) == T "Il vettore dones deve avere lunghezza T"
    advantages = zeros(Float64, T)
    gae = 0.0
    for t in T:-1:1
        delta = rewards[t] + γ * values[t+1] * (1.0 - dones[t]) - values[t]
        gae = delta + γ * λ * (1.0 - dones[t]) * gae
        advantages[t] = gae
    end
    return advantages
end

function update_policy!(agent::PPOAgent, bootstrap_by_env::Vector{Float64})
    T = _min_len(agent.buffer)
    if T == 0; return; end
    _trim_to!(agent.buffer, T)

    r   = agent.buffer.rewards
    v   = agent.buffer.values
    d   = agent.buffer.dones
    eid = agent.buffer.env_ids

    adv      = zeros(Float64, T)
    last_v   = Dict{Int,Float64}(i => bootstrap_by_env[i] for i in 1:length(bootstrap_by_env))
    last_adv = Dict{Int,Float64}(i => 0.0 for i in 1:length(bootstrap_by_env))

    γ, λ = agent.gamma, agent.lambda
    @inbounds for t in T:-1:1
        e = eid[t]
        next_v = d[t] ? 0.0 : get(last_v, e, 0.0)
        δ = r[t] + γ * next_v - v[t]
        adv[t] = δ + γ * λ * (d[t] ? 0.0 : get(last_adv, e, 0.0))
        last_adv[e] = adv[t]
        last_v[e]   = v[t]
        if d[t]; last_adv[e] = 0.0; last_v[e] = 0.0; end
    end
    returns = adv .+ v

    μ, σ = mean(adv), std(adv) + 1e-8
    norm_adv = (adv .- μ) ./ σ

    data_states     = agent.buffer.states
    data_actions    = agent.buffer.actions
    data_returns    = returns
    data_advantages = norm_adv
    data_old_logps  = agent.buffer.log_probs

    N = length(data_states)
    if N == 0; return; end

    target_KL = 0.005

    for _ in 1:agent.n_update_epochs
        idx = randperm(agent.rng, N)
        running_kl = 0.0; num_mb = 0
        for start in 1:agent.mini_batch_size:length(idx)
            stop = min(start + agent.mini_batch_size - 1, length(idx))
            bi = idx[start:stop]

            batch_states     = hcat(data_states[bi]...)
            batch_actions    = hcat(data_actions[bi]...)
            batch_returns    = data_returns[bi]
            batch_advantages = data_advantages[bi]
            batch_old_logps  = data_old_logps[bi]

            grads_actor_tuple = Flux.gradient(agent.policy.actor) do actor_model
                dist      = actor_model(batch_states)
                new_logps = logprob_squashed(dist, batch_actions)
                ratios    = exp.(new_logps .- batch_old_logps)
                surr1 = ratios .* batch_advantages
                surr2 = clamp.(ratios, 1 - agent.clip_range, 1 + agent.clip_range) .* batch_advantages
                actor_loss = -mean(min.(surr1, surr2))
                ent = sum(entropy.(dist), dims=1)[:]
                entropy_term = -mean(ent)
                actor_loss + agent.entropy_loss_weight * entropy_term
            end
            grads_actor = first(grads_actor_tuple)
            new_actor_opt_state, new_actor =
                Flux.update!(agent.actor_opt_state, agent.policy.actor, grads_actor)
            agent.actor_opt_state = new_actor_opt_state
            agent.policy = PPOPolicy(new_actor, agent.policy.critic)

            grads_critic_tuple = Flux.gradient(agent.policy.critic) do critic_model
                v̂ = critic_model(batch_states)[:]
                Flux.mse(v̂, batch_returns) * agent.critic_loss_weight
            end
            grads_critic = first(grads_critic_tuple)
            new_critic_opt_state, new_critic =
                Flux.update!(agent.critic_opt_state, agent.policy.critic, grads_critic)
            agent.critic_opt_state = new_critic_opt_state
            agent.policy = PPOPolicy(agent.policy.actor, new_critic)

            dist_after  = agent.policy.actor(batch_states)
            new_logps   = logprob_squashed(dist_after, batch_actions)
            running_kl += mean(batch_old_logps .- new_logps)
            num_mb += 1
        end
        if num_mb > 0 && running_kl / num_mb > target_KL
            break
        end
    end
end


