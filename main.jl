using Distributed

# 1. Aggiungi i processi (Solo sul Master)
if nprocs() == 1
    addprocs(16) 
end


@everywhere begin
    using PyCall
    # Aggiungi il path per trovare il file python
    pushfirst!(PyVector(pyimport("sys")["path"]), "")
    
    # Inizializza il modulo python globalmente su ogni worker
    if !@isdefined(qw)
        Core.eval(Main, :(const qw = pyimport("quantum_well")))
    end
end


@everywhere include("RL.jl")

@everywhere begin



    # fissi ky e delta 
    PARAMS_MIN = [0.3,   1500,       1499,   -0.001, 0.33  ]  # [a, k, ky, delta, d]
    PARAMS_MAX = [0.9, 3000.0,   1500.1,   0.001,    3  ]


    INITIAL_P  = [0.3, 1500,  1500 ,  0.0, 0.33]

    Nx, Ny   = 20, 20
    x0, xL   = -1.0,  1.0
    y0, yL   = -0.5,  0.5
    kappa    = 2326.0    # Coulomb strength (Mott-Hubbard crossover, U/t ~ 1)
    epsilon  = 0.01 
    Nstates  = 200   
    N_CI_COMPUTE  = 200   # raised from 40; covers L0-L4 x R0-R4
    SMOOTH_PL = false
    M_LOC     =32     # localisation subspace size (try 8, 12, 16, ...)
    X_CUT     = 0.0       # dividing surface between wells (barrier centre)
    SIGMA_PL  = 0.03      # SP orbitals to retain

    #DVR
    x_grid, y_grid, w_x, w_y, T2D = qw.build_2d_dvr(Nx, Ny, x0, xL, y0, yL)

    #Kernel Coulomb
    K = qw.precompute_coulomb_kernel(x_grid, y_grid, w_x, w_y, kappa, epsilon)

end



## Fuori dal Loop di training :


state_dim = 5
action_dim = 5

n_envs=1;
envs = [WellEnv(x_grid, y_grid, T2D, K; max_steps=1) for _ in 1:n_envs];



actor = Actor(state_dim, action_dim);
critic  = Critic(state_dim);

agent = PPOAgent(
    actor, 
    critic, 
    n_env = n_envs,              # 1 per ogni worker
    n_rollout = 128,         # Quanti step raccogliere prima di aggiornare la rete
    mini_batch_size = 32,
    actor_optimizer = Adam(1f-4),
    critic_optimizer = Adam(1f-3),
    entropy_loss_weight = 0.02
);



##  TRAINING LOOP
using Printf

n_iterations = 100    # Quante volte aggiornare la rete
best_reward = -Inf32   # Per tracciare il record assoluto
best_params = copy(INITIAL_P)

println("🚀 Avvio training parallelo su $n_envs worker...")

for iter in 1:n_iterations
    # 1. Reset e Check dello Stato Iniziale
    states = [RLBase.reset!(env) for env in envs]
    
    # DEBUG: Verifica se il reset produce NaN
    for (i, st) in enumerate(states)
        if any(isnan, st)
            @error "NaN rilevato nel RESET dell'ambiente $i"
            println("Parametri env: ", envs[i].params)
            println("PARAMS_MIN: ", PARAMS_MIN)
            println("PARAMS_MAX: ", PARAMS_MAX)
            error("Lo stato iniziale è corrotto. Controlla la divisione in RLBase.state.")
        end
    end

    iteration_rewards = Float32[]
    steps_to_collect = agent.n_rollout ÷ n_envs
    
    for s_idx in 1:steps_to_collect
        # 2. Check Pesi della Rete prima di select_action
        # Se i pesi sono NaN, select_action restituirà azioni 0 a causa del nostro guardrail
        if any(isnan, Flux.params(agent.policy.actor)[1])
            @error "I PESI DELL'ACTOR SONO DIVENTATI NaN!"
            error("Training interrotto: la rete è esplosa.")
        end

        # Selezione azioni
        actions_info = [select_action(agent, st) for st in states]
        
        actions     = [a[1] for a in actions_info] 
        logps       = [a[2] for a in actions_info]
        values      = [a[3] for a in actions_info]
        raw_actions = [a[4] for a in actions_info] 

        # 3. Debug sulle azioni (Se vedi solo 0, qui capiamo perché)
        if iter == 1 && s_idx == 1
            println("--- Debug Primo Step ---")
            println("Esempio Azione Squashed ([-1,1]): ", actions[1])
            println("Esempio Azione Raw (u): ", raw_actions[1])
            println("Esempio Valore Critic: ", values[1])
        end

        # STEP PARALLELO
        next_states, rewards, dones = step_envs!(envs, actions)

        # 4. Debug sui Reward
        if any(isnan, rewards)
            idx_nan = findfirst(isnan, rewards)
            @error "NaN rilevato nel REWARD dell'ambiente $idx_nan!"
            println("Azione inviata: ", actions[idx_nan])
            error("La simulazione fisica ha restituito NaN.")
        end

        for i in 1:n_envs
            store_transition!(agent, states[i], raw_actions[i], rewards[i], dones[i], logps[i], values[i], env_id=i)
            push!(iteration_rewards, rewards[i])

            if rewards[i] > best_reward
                global best_reward = rewards[i]
                global best_params = copy(envs[i].params)
                @printf("\n🌟 NUOVO RECORD! Reward: %.6f | Params: %s\n", best_reward, string(round.(best_params, digits=3)))
            end
        end
        
        states = next_states
    end

    # 3. Aggiornamento PPO
    last_values = Float32[only(agent.policy.critic(st)) for st in states]
    
    # DEBUG: Verifica bootstrap prima dell'update
    if any(isnan, last_values)
        @error "Il Critic ha restituito NaN per l'ultimo stato!"
        error("Update annullato.")
    end

    update!(agent, bootstrap_values_by_env=last_values)

    # 4. Logging
    if iter % 5 == 0 || iter == 1
        avg_iter_reward = mean(iteration_rewards)
        @printf("Iterazione %4d | Reward Medio: %.4f | Best Reward: %.4f\n", 
                iter, avg_iter_reward, best_reward)
    end
end

println("\n🏁 Training terminato!")
println("🏆 Miglior Accessible Entropy trovata: ", best_reward)
println("📍 Parametri ottimali: ", best_params)

## ANALISI
using PyPlot
pygui(true)




