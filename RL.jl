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

###  model aware RL ci andrebbe anche l`info sulle energie come stato, per ora solo params 
### 27/04: ho messo anche le variabili nello stato, vediamo che succede


"""

BOH DOPO POSSO AGGIUNGERE ANCHE QUESTO PER ORA NO
model-aware RL : rimane sewmpre model-free perchè PPO basa l`ottimizzazione 
del gradiente solo sui pesi della rete neurale (objective f). è però 
model-aware perchè nello stato ci sto passando delle osservabili fisiche che 
dovrebbero aiutare la convergenza del modello (?) e anche a rispettare delle
regole (?) - questo da fare check -  energy gap & the prob p1? Se aggiungo
info sullo stato devo fare mini-sper nell`episodio

(!) *** MI DEVO RICORDARE CHE DEVO NORMALIZZARE TRA -1 E 1 ANCHE LE OSSERVABILI SULLO STATO 
CHE STAI AGGIUNGENDO ALLO STATO ***



massimizzano anche il gap tra singoletto e tripletto -> fallo anche tu

gap = abs(E2[2] - E2[1]) # Differenza tra i primi due stati CI
env.reward = (E_acc * p1) + 0.1 * gap



"""

mutable struct WellEnv <: AbstractEnv
    #lo stato visto da RL sono i parametri correnti [a, k, d, delta, ky]
    #model-aware: aggiungo p1 e energy gap ?
    params::Vector{Float32} 


    x_grid::Array{Float32,1}
    y_grid::Array{Float32,1}
    T2D::Array{Float32,2}
    K::Array{Float32,2}


    reward::Float32

    done::Bool
    steps::Int
    max_steps::Int
    rng::StableRNG
end
 

function WellEnv(x_grid, y_grid, T2D, K; max_steps=1)
    
    return WellEnv(
        copy(INITIAL_P), 
        x_grid, 
        y_grid, 
        T2D, 
        K, 
        0.0,    # reward
        false,  # done
        0,      # steps
        max_steps, 
        StableRNG(123)
        )
end

function RLBase.action_space(env::WellEnv)
    return [ClosedInterval(-1.0, 1.0) for _ in 1:5]
end



function RLBase.state(env::WellEnv)
    # Riscalamento lineare: [min, max] -> [-1, 1]
    # il broadcasting (@.) serve per operare su tutto il vettore
    s = @. 2.0 * (env.params - PARAMS_MIN) / (PARAMS_MAX - PARAMS_MIN) - 1.0
    
    
    return Float32.(s)
end

RLBase.state_space(env::WellEnv) = [ClosedInterval(-1.0f0, 1.0f0) for _ in 1:5]
RLBase.action_space(env::WellEnv) = [ClosedInterval(-1.0f0, 1.0f0) for _ in 1:5]


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

    init_ortho = Flux.orthogonal
    
    chain = Chain(
        # Strati nascosti
        Dense(state_dim => 128, tanh, init = init_ortho),
        Dense(128 => 64, tanh, init = init_ortho),
        Dense(64 => 32, tanh, init = init_ortho),
        
        # Strato di output: usiamo un guadagno (gain) molto piccolo (0.01)
        # Questo mantiene le azioni iniziali vicine allo zero e previene i NaN
        Dense(32 => action_dim * 2, init = Flux.orthogonal(gain=0.01))
    )
    Actor(chain)
end

function (actor::Actor)(state)
    x = actor.chain(state)
    
    # Check preventivo (se x è già NaN, il gradiente è già rotto altrove)
    if any(isnan, x)
        @warn "Actor output is NaN"
        x = fill!(similar(x), 0.0f0)
    end

    action_dim = size(x, 1) ÷ 2
    
    # Estraiamo mu e log_sigma
    if ndims(x) == 1
        μ = x[1:action_dim]
        logσ = x[action_dim+1:end]
    else
        μ = x[1:action_dim, :]
        logσ = x[action_dim+1:end, :]
    end

    # STABILIZZAZIONE CRITICA:
    # 1. Non applichiamo tanh qui! Lo faremo dopo il campionamento.
    # 2. Stringiamo il clamp su logσ. -20 è il limite standard per evitare underflow, 
    #    mentre 2.0 evita esplosioni (exp(2) ≈ 7.3).
    logσ = clamp.(logσ, -20.0f0, 2.0f0)
    σ = exp.(logσ)

    # Restituiamo la distribuzione sui valori "unbounded"
    return Normal.(μ, σ)
end


struct Critic
    chain::Chain
end
Flux.@layer Critic

function Critic(state_dim::Int)
    chain = Flux.Chain(
       Flux.Dense(state_dim, 128, tanh),
        Flux.Dense(128, 64, tanh),
       Flux.Dense(64, 32, tanh),
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
        # USA IL PUNTO davanti al meno: 1.0f0 .- a.^2
        lp = sum(log_base) - sum(log.(1.0f0 .- a.^2 .+ 1f-6))
        return a, lp, u
    else
        # USA IL PUNTO davanti al meno: 1.0f0 .- a.^2
        lp = vec(sum(log_base, dims=1) .- sum(log.(1.0f0 .- a.^2 .+ 1f-6), dims=1))
        return a, lp, u
    end
end

function logprob_squashed(dist, raw_actions)
    # raw_actions sono i valori 'u' salvati nel buffer (prima del tanh)
    a = tanh.(raw_actions)
    lp = sum(logpdf.(dist, raw_actions), dims=1) .- sum(log.(1.0f0 .- a.^2 .+ 1f-6), dims=1)
    return vec(lp)
end

function step!(env::WellEnv, action::AbstractVector)

    ## take the action from RL and build the params for the potential V
    # mini-variations in the step for the actions (?) instead of a 1 step fopr each episode
    action = Float32.(action)
    # action ∈ [-1, 1] -> params ∈ [MIN, MAX]
    env.params .= PARAMS_MIN .+ (action .+ 1.0f0) .* 0.5f0 .* (PARAMS_MAX .- PARAMS_MIN)
    env.params = clamp.(env.params, PARAMS_MIN, PARAMS_MAX)

    
    # da julia a python lo fa PyCall (?) vedi se funziona - pare di si 
    params_py = Dict(
            "a"     => env.params[1],
            "k"     => env.params[2],
            "ky"    =>  1500.0, #env.params[3],
            "delta" =>  0,      #env.params[4]
            "d"     => env.params[5]  
        )


    xg_py = vec(collect(env.x_grid))
    yg_py = vec(collect(env.y_grid))

   
    V2D = qw.build_potential_matrix(xg_py, yg_py, params_py)
    

    #= 
    Diagonalising single-particle Hamiltonian
    =#


    single_res = qw.single_particle_eigenstates(T2D, V2D, Nstates)
    single_energies, single_vecs = single_res[1], single_res[2]



    U_loc, vecs_loc, lam_loc, lr_labels, S_ortho = qw.localise_orbitals_projector_DVR(
        single_vecs, x_grid, y_grid,
        M=M_LOC, x_cut=X_CUT, smooth=SMOOTH_PL, sigma=SIGMA_PL
    )

    #=
    
    Vincoli sui guess dei parametri dati da RL:

    FPM email -> non contare stati in cui non abbiamo autovalori circa 0 o circa 1 ,
    quindi se l autovalore è 0.5 gli stati non sono localizzati

    =#
        
    n_ambiguous = count(l -> abs(l - 0.5) <= 0.05, lam_loc)
    if n_ambiguous > 5
        env.reward = 0
        env.done = true
        
    end
    


    ## CI Interaction
    
    slater_basis = qw.build_slater_basis_sorted(Nstates, single_energies)
    n_compute    = min(N_CI_COMPUTE, length(slater_basis))


    H_slater     = qw.build_ci_hamiltonian(slater_basis, single_energies,
                                        single_vecs, K, n_compute)

    # stato a due elettroni
    E2, C2 = eigen(H_slater)




    C2_spin_purified = qw.purify_degenerate_spin_subspaces(
    E2, C2, slater_basis[1:n_compute], energy_tol=1e-6, spin_tol=1e-8
    )


    gap = E2[2] - E2[1]

    n_state_py = 0
    n_state_jl = 1


    mode_tags_spin = repeat(lr_labels, inner=2)

    S2, Sz, _ = qw.compute_spin_and_entanglement(
        C2_spin_purified[:, n_state_jl], slater_basis[1:n_compute], Nstates
    )


    Om_full = qw.ci_to_spinorbital_Omega(C2_spin_purified[:, n_state_jl], slater_basis[1:n_compute], Nstates)
    Om_sub = qw.truncate_Omega_to_subspace(Om_full, M_LOC)



    U_spin = qw.spinorbital_U_from_spatial(U_loc) 

    U_conj = conj(U_spin)

    Om_loc = U_conj' * Om_sub * U_conj
    
    
    Om_loc = 0.5 * (Om_loc - transpose(Om_loc))

   
    rho2_loc, pairs_spin = qw.Omega_to_rho2_pair(Om_loc)
    rhoL0, rhoL1, rhoL2, _ = qw.rhoL_from_rho2_pairs_spin(rho2_loc, pairs_spin, mode_tags_spin)

    
    E_acc, (p0,p1,p2) = qw.accessible_entropy(rhoL0, rhoL1, rhoL2)
    env.reward = E_acc #+ 0.1 * gap
    
    
    env.done = true

    return env.reward, env.done
end

function step_envs!(envs::Vector{WellEnv}, actions::Vector)
    # Usa pmap per distribuire il calcolo su processi diversi
    results = pmap(1:length(envs)) do i
        r, d = step!(envs[i], actions[i])
        s = RLBase.state(envs[i])
        (s, Float32(r), d)
    end
    
    # "Scompatta" i risultati
    states = [res[1] for res in results]
    rewards = [res[2] for res in results]
    dones = [res[3] for res in results]
    
    return states, rewards, dones
end

mutable struct PPOPolicy
    actor::Actor
    critic::Critic
end
Functors.@functor PPOPolicy

mutable struct PPOBuffer
    states::Vector{Vector{Float32}}
    actions::Vector{Vector{Float32}}
    rewards::Vector{Float32}
    dones::Vector{Bool}
    log_probs::Vector{Float32}
    values::Vector{Float32}
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
    gamma::Float32
    lambda::Float32
    clip_range::Float32
    entropy_loss_weight::Float32
    critic_loss_weight::Float32
    max_grad_norm::Float32
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

function select_action(agent::PPOAgent, state::Vector{Float32})
    dists = agent.policy.actor(state)
    
    # Otteniamo azione squashed (a), logprob (lp) e azione raw (u)
    a, lp, u = sample_squashed(dists)
    
    value = agent.policy.critic(state)

    # Nota: nel buffer dovresti spingere 'u' invece di 'a' 
    # per una stabilità totale durante l'update!
    return collect(a), lp, value, collect(u) 
end
function select_action_eval(agent::PPOAgent, state::Vector{Float32})
    dists = agent.policy.actor(state)
    # μ è il valore raw centrale
    μ_raw = [d.μ for d in dists]
    return tanh.(μ_raw) # Azione deterministica pulita
end
function ready_to_update(agent::PPOAgent)
    return length(agent.buffer.rewards) >= agent.n_rollout
end

function update!(agent::PPOAgent; bootstrap_values_by_env::Vector{Float32})
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
    agent.buffer.rewards = Float32[]
    agent.buffer.dones = Bool[]
    agent.buffer.values = Float32[]
    agent.buffer.log_probs = Float32[]
end

function compute_gae(rewards::Vector{Float32}, values::Vector{Float32}, dones::Vector{Bool}; γ=0.99, λ=0.95)
    T = length(rewards)
    @assert length(values) == T + 1 "Il vettore values deve avere lunghezza T+1"
    @assert length(dones) == T "Il vettore dones deve avere lunghezza T"
    advantages = zeros(Float32, T)
    gae = 0.0
    for t in T:-1:1
        delta = rewards[t] + γ * values[t+1] * (1.0 - dones[t]) - values[t]
        gae = delta + γ * λ * (1.0 - dones[t]) * gae
        advantages[t] = gae
    end
    return advantages
end

function update_policy!(agent::PPOAgent, bootstrap_by_env::Vector{Float32})
    # 1. Verifica e sincronizzazione del buffer
    T = _min_len(agent.buffer)
    if T == 0; return; end
    _trim_to!(agent.buffer, T)

    r   = agent.buffer.rewards
    v   = agent.buffer.values
    d   = agent.buffer.dones
    eid = agent.buffer.env_ids

    # 2. Calcolo dei vantaggi (GAE - Generalized Advantage Estimation)
    adv      = zeros(Float32, T)
    # bootstrap_by_env contiene V(s_last) per ogni ambiente parallelo
    last_v   = Dict{Int,Float32}(i => bootstrap_by_env[i] for i in 1:length(bootstrap_by_env))
    last_adv = Dict{Int,Float32}(i => 0.0f0 for i in 1:length(bootstrap_by_env))

    γ, λ = agent.gamma, agent.lambda
    @inbounds for t in T:-1:1
        e = eid[t]
        next_v = d[t] ? 0.0f0 : get(last_v, e, 0.0f0)
        δ = r[t] + γ * next_v - v[t]
        adv[t] = δ + γ * λ * (d[t] ? 0.0f0 : get(last_adv, e, 0.0f0))
        last_adv[e] = adv[t]
        last_v[e]   = v[t]
        if d[t]; last_adv[e] = 0.0f0; last_v[e] = 0.0f0; end
    end
    returns = adv .+ v

    # Normalizzazione degli Advantages
    μ_adv, σ_adv = mean(adv), std(adv) + 1f-8
    norm_adv = (adv .- μ_adv) ./ σ_adv

    data_states     = agent.buffer.states
    data_actions    = agent.buffer.actions
    data_returns    = returns
    data_advantages = norm_adv
    data_old_logps  = agent.buffer.log_probs

    N = length(data_states)
    target_KL = 0.015f0 

    # 3. Loop di Ottimizzazione (Epochs)
    for epoch in 1:agent.n_update_epochs
        idx = randperm(agent.rng, N)
        running_kl = 0.0f0
        num_mb = 0
        
        for start in 1:agent.mini_batch_size:length(idx)
            stop = min(start + agent.mini_batch_size - 1, length(idx))
            bi = idx[start:stop]

            # Mini-batching
            batch_states     = hcat(data_states[bi]...)
            batch_actions    = hcat(data_actions[bi]...)
            batch_returns    = data_returns[bi]
            batch_advantages = data_advantages[bi]
            batch_old_logps  = data_old_logps[bi]

            # --- UPDATE ACTOR ---
            val_a, grads_a_tuple = Flux.withgradient(agent.policy.actor) do actor_model
                dist      = actor_model(batch_states)
                new_logps = logprob_squashed(dist, batch_actions)
                ratios    = exp.(new_logps .- batch_old_logps)
                
                surr1 = ratios .* batch_advantages
                surr2 = clamp.(ratios, 1.0f0 - agent.clip_range, 1.0f0 + agent.clip_range) .* batch_advantages
                
                actor_loss = -mean(min.(surr1, surr2))
                ent = sum(entropy.(dist), dims=1)[:]
                return actor_loss - agent.entropy_loss_weight * mean(ent)
            end
            grads_a = grads_a_tuple[1]

            # Clipping Global Norm sicuro con destructure
            flat_grads_a, _ = Flux.destructure(grads_a)
            gnorm_a = sqrt(sum(abs2, flat_grads_a))
            if gnorm_a > agent.max_grad_norm
                scale = agent.max_grad_norm / (gnorm_a + 1f-6)
                grads_a = fmap(x -> x isa AbstractArray ? x .* scale : x, grads_a)
            end
            
            # Applicazione gradienti Actor
            agent.actor_opt_state, agent.policy.actor = Flux.update!(agent.actor_opt_state, agent.policy.actor, grads_a)

            # --- UPDATE CRITIC ---
            val_c, grads_c_tuple = Flux.withgradient(agent.policy.critic) do critic_model
                v_hat = critic_model(batch_states)
                return Flux.mse(vec(v_hat), batch_returns) * agent.critic_loss_weight
            end
            grads_c = grads_c_tuple[1]

            # Clipping Global Norm sicuro con destructure
            flat_grads_c, _ = Flux.destructure(grads_c)
            gnorm_c = sqrt(sum(abs2, flat_grads_c))
            if gnorm_c > agent.max_grad_norm
                scale = agent.max_grad_norm / (gnorm_c + 1f-6)
                grads_c = fmap(x -> x isa AbstractArray ? x .* scale : x, grads_c)
            end

            # Applicazione gradienti Critic
            agent.critic_opt_state, agent.policy.critic = Flux.update!(agent.critic_opt_state, agent.policy.critic, grads_c)

            # Monitoraggio KL Divergence
            dist_after  = agent.policy.actor(batch_states)
            new_logps   = logprob_squashed(dist_after, batch_actions)
            running_kl += mean(batch_old_logps .- new_logps)
            num_mb += 1
        end
        
        # Early stopping se la politica cambia troppo
        if num_mb > 0 && (running_kl / num_mb) > target_KL
            break
        end
    end
end