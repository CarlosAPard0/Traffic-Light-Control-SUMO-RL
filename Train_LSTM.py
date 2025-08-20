import os          
import sys          
import ray          
from ray import tune          
from ray.rllib.algorithms.ppo import PPOConfig          
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv          
from ray.tune.registry import register_env          
import supersuit as ss          
from ray.tune.schedulers import ASHAScheduler          
        
if "SUMO_HOME" in os.environ:          
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")          
    sys.path.append(tools)          
else:          
    sys.exit("Please declare the environment variable 'SUMO_HOME'")          
          
import sumo_rl          
        
if __name__ == "__main__":          
    ray.init(num_cpus=32, num_gpus=1, ignore_reinit_error=True)          
    print(f"Ray version: {ray.__version__}")          
              
    env_name = "PPO"    
              
    def make_env():            
        def advanced_reward_function(traffic_signal):          
            waiting_time_reward = traffic_signal._diff_waiting_time_reward()          
            speed_reward = traffic_signal._average_speed_reward() * 0.1          
            queue_penalty = traffic_signal._queue_reward() * 0.05          
            pressure_reward = traffic_signal._pressure_reward() * 0.02          
            phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05          
            return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus          
              
        env = sumo_rl.parallel_env(            
            net_file="Data/network.net.xml",          
            route_file="Data/routes.rou.xml",         
            out_csv_name="Outputs/ppo_advanced",            
            use_gui=False,            
            reward_fn=advanced_reward_function,    
            num_seconds=540,        
            begin_time=0,        
            time_to_teleport=360,        
        )            
        env = ss.pad_observations_v0(env)      
        env = ss.frame_stack_v1(env, 3)    
        return env    
              
    register_env(env_name, lambda _: ParallelPettingZooEnv(make_env()))          
              
    config = (            
        PPOConfig()            
        .environment(env=env_name, disable_env_checking=True)            
        .rollouts(num_rollout_workers=16, rollout_fragment_length="auto")    
        .training(            
            train_batch_size=256,            
            lr=5e-4,            
            gamma=0.99,            
            lambda_=0.95,            
            use_gae=True,            
            clip_param=0.2,            
            entropy_coeff=0.01,            
            vf_loss_coeff=0.5,            
            sgd_minibatch_size=128,            
            num_sgd_iter=15,            
            model={              
                # Configuración LSTM  
                "use_lstm": True,  
                "lstm_cell_size": 128,  
                "lstm_use_prev_action": True,  
                "lstm_use_prev_reward": True,  
                  
                # Capas densas  
                "fcnet_hiddens": [256, 128],      
                "fcnet_activation": "swish",      
                "vf_share_layers": True,      
            }           
        )            
        .framework(framework="torch")            
        .resources(num_gpus=1, num_cpus_per_worker=1)            
    )      
        
    scheduler = ASHAScheduler(          
        metric="episode_reward_mean",          
        mode="max",          
        max_t=25000,          
        grace_period=20,          
        reduction_factor=2          
    )          
              
    print("Iniciando entrenamiento multi-agente con LSTM...")          
              
    tune.run(    
        "PPO",    
        name="PPO_LSTM_0",    
        stop={"timesteps_total": 10000000},    
        checkpoint_freq=5,   
        keep_checkpoints_num=10,    
        checkpoint_score_attr="episode_reward_mean",    
        local_dir=os.path.abspath("Outputs/ray_results"),    
        config=config.to_dict(),    
        scheduler=scheduler,    
        verbose=2,    
    )  
              
    print("Entrenamiento completado!")          
    ray.shutdown()