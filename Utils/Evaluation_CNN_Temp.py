import os  
import sys  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import ray  
from ray.rllib.algorithms.algorithm import Algorithm  
from ray.tune.registry import register_env  
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv  
import supersuit as ss  
from scipy import stats  

# Configure SUMO_HOME  
if "SUMO_HOME" in os.environ:  
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")  
    sys.path.append(tools)  
else:  
    sys.exit("Please declare the environment variable 'SUMO_HOME'")  

import sumo_rl  

# --- Configuration ---  
USE_GUI = False  
CHECKPOINT_PATH = "../Outputs/ray_results/PPO_CNN_0/PPO_PPO_77c16_00000_0_2025-08-20_19-32-54/checkpoint_000091--81"  
OUTPUT_DIR = "../Results/CNN"  
SIMULATION_TIME = 540  # 9 minutes  
DELTA_TIME = 5  
NUM_SCENARIOS = 10  

# Matplotlib config  
plt.rcParams.update({  
    'figure.figsize': (14, 8),  
    'font.size': 18,  
    'axes.titlesize': 20,  
    'axes.labelsize': 18,  
    'xtick.labelsize': 16,  
    'ytick.labelsize': 16,  
    'legend.fontsize': 16,  
    'lines.linewidth': 2.5,  
    'grid.alpha': 0.3,  
    'axes.grid': True,  
    'figure.autolayout': True  
})  

COLORS = {  
    'Traditional': '#d62728',  
    'AI (PPO-CNN)': '#1f77b4'  
}  

def create_output_directory():  
    """Create output directory"""  
    os.makedirs(OUTPUT_DIR, exist_ok=True)  
    return OUTPUT_DIR  

def create_traditional_env(route_file):  
    """Create environment with traditional traffic lights"""  
    env = sumo_rl.SumoEnvironment(  
        net_file="../Data/network.net.xml",  
        route_file=route_file,  
        use_gui=USE_GUI,  
        num_seconds=SIMULATION_TIME,  
        delta_time=DELTA_TIME,  
        begin_time=0,  
        time_to_teleport=360,  
        add_system_info=True,  
        add_per_agent_info=True,  
        fixed_ts=True,  # Traditional traffic lights  
        single_agent=False  
    )  
    return env  

def create_ai_env(route_file):  
    """Create environment for the CNN model"""  
    def advanced_reward_function(traffic_signal):  
        waiting_time_reward = traffic_signal._diff_waiting_time_reward()  
        speed_reward = traffic_signal._average_speed_reward() * 0.1  
        queue_penalty = traffic_signal._queue_reward() * 0.05  
        pressure_reward = traffic_signal._pressure_reward() * 0.02  
        phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05  
        return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus  

    env = sumo_rl.parallel_env(  
        net_file="../Data/network.net.xml",  
        route_file=route_file,  
        use_gui=USE_GUI,  
        reward_fn=advanced_reward_function,  
        num_seconds=SIMULATION_TIME,  
        delta_time=DELTA_TIME,  
        begin_time=0,  
        time_to_teleport=360,  
        add_system_info=True,  
        add_per_agent_info=True,  
    )  
    env = ss.pad_observations_v0(env)  
    env = ss.frame_stack_v1(env, 3)  
    return env  

def collect_step_metrics(info_or_infos, simulation_time):      
    """Extracts step metrics with simulation time"""      
    metrics = {}      
          
    if isinstance(info_or_infos, dict):      
        if 'system_mean_waiting_time' in info_or_infos:      
            info = info_or_infos      
        else:      
            info = None      
            for agent_id, agent_info in info_or_infos.items():      
                if 'system_mean_waiting_time' in agent_info:      
                    info = agent_info      
                    break      
              
        if info:      
            metrics = {      
                'simulation_time': simulation_time,      
                'system_mean_waiting_time': info.get('system_mean_waiting_time', 0),      
                'system_mean_speed': info.get('system_mean_speed', 0),      
                'system_total_stopped': info.get('system_total_stopped', 0),      
                'system_total_running': info.get('system_total_running', 0),      
                'system_total_arrived': info.get('system_total_arrived', 0),      
                'system_total_departed': info.get('system_total_departed', 0),      
                'system_total_waiting_time': info.get('system_total_waiting_time', 0),      
            }      
          
    return metrics  

def run_traditional_scenarios():  
    """Run evaluation with traditional traffic lights"""  
    print("Evaluating traditional traffic lights...")  
    all_metrics = []  
    
    for scenario in range(1, NUM_SCENARIOS + 1):  
        route_file = f"../Data/Routes_Test/routes{scenario}.rou.xml"  
        print(f"  Scenario {scenario}: {route_file}")  
          
        env = create_traditional_env(route_file)  
        scenario_metrics = []  
          
        observations = env.reset()  
        done = {"__all__": False}  
          
        while not done["__all__"]:  
            observations, rewards, done, info = env.step({})  
              
            current_time = env.sim_step  
            step_metrics = collect_step_metrics(info, current_time)  
            if step_metrics:  
                step_metrics['model_type'] = 'Traditional'  
                step_metrics['scenario'] = scenario  
                scenario_metrics.append(step_metrics)  
          
        env.close()  
        all_metrics.extend(scenario_metrics)  
        print(f"    ✓ Completed: {len(scenario_metrics)} timesteps")  
    
    return all_metrics  

def run_ai_scenarios():    
    """Run evaluation with CNN model"""    
    print("Evaluating CNN model...")    
    ray.init(ignore_reinit_error=True)    
        
    def create_registered_env(_):    
        def advanced_reward_function(traffic_signal):    
            waiting_time_reward = traffic_signal._diff_waiting_time_reward()    
            speed_reward = traffic_signal._average_speed_reward() * 0.1    
            queue_penalty = traffic_signal._queue_reward() * 0.05    
            pressure_reward = traffic_signal._pressure_reward() * 0.02    
            phase_stability_bonus = 0.1 if traffic_signal.time_since_last_phase_change > traffic_signal.min_green else -0.05    
            return waiting_time_reward + speed_reward + queue_penalty + pressure_reward + phase_stability_bonus    
            
        env = sumo_rl.parallel_env(    
            net_file="../Data/network.net.xml",    
            route_file="../Data/routes.rou.xml",    
            use_gui=USE_GUI,    
            reward_fn=advanced_reward_function,    
            num_seconds=SIMULATION_TIME,    
            delta_time=DELTA_TIME,    
            begin_time=0,    
            time_to_teleport=360,    
            add_system_info=True,    
            add_per_agent_info=True,    
        )    
        env = ss.pad_observations_v0(env)    
        env = ss.frame_stack_v1(env, 3)    
        return ParallelPettingZooEnv(env)    
        
    register_env("PPO", create_registered_env)    
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)    
        
    all_metrics = []    
        
    for scenario in range(1, NUM_SCENARIOS + 1):    
        route_file = f"../Data/Routes_Test/routes{scenario}.rou.xml"    
        print(f"  Scenario {scenario}: {route_file}")    
            
        env = create_ai_env(route_file)    
        scenario_metrics = []    
            
        reset_result = env.reset()    
        if isinstance(reset_result, tuple):    
            obs_dict, infos = reset_result    
        else:    
            obs_dict = reset_result    
            
        done = False    
            
        while not done:    
            actions = {}    
            for agent_id, agent_obs in obs_dict.items():    
                action = algo.compute_single_action(    
                    observation=agent_obs,    
                    policy_id="default_policy",    
                )    
                actions[agent_id] = action    
                
            obs_dict, rewards, terminations, truncations, infos = env.step(actions)    
                
            current_time = env.unwrapped.env.sim_step    
            step_metrics = collect_step_metrics(infos, current_time)    
            if step_metrics:    
                step_metrics['model_type'] = 'AI (PPO-CNN)'    
                step_metrics['scenario'] = scenario    
                scenario_metrics.append(step_metrics)    
                
            done = all(terminations.values()) or all(truncations.values())    
            
        env.close()    
        all_metrics.extend(scenario_metrics)    
        print(f"    ✓ Completed: {len(scenario_metrics)} timesteps")    
        
    ray.shutdown()    
    return all_metrics  

def plot_temporal_evolution_with_ci(df, metric_name, ylabel, title, save_path):    
    """Generate temporal plot with 95% confidence intervals"""    
    plt.figure(figsize=(14, 8))    
    
    scenario_time_summaries = df.groupby(['scenario', 'model_type', 'simulation_time']).agg({      
        'system_mean_speed': 'mean',      
        'system_mean_waiting_time': 'mean'  
    }).reset_index()  
    
    for model_type in scenario_time_summaries['model_type'].unique():    
        model_data = scenario_time_summaries[scenario_time_summaries['model_type'] == model_type]    
        time_stats = []    
        for time_point in sorted(model_data['simulation_time'].unique()):    
            time_data = model_data[model_data['simulation_time'] == time_point]    
            values = time_data[metric_name].values    
                
            if len(values) > 1:    
                mean_val = np.mean(values)    
                std_err = stats.sem(values)    
                ci = std_err * stats.t.ppf(0.975, len(values)-1)    
                    
                time_stats.append({    
                    'time': time_point,    
                    'mean': mean_val,    
                    'ci_lower': mean_val - ci,    
                    'ci_upper': mean_val + ci    
                })    
            else:    
                val = values[0] if len(values) > 0 else 0    
                time_stats.append({    
                    'time': time_point,    
                    'mean': val,    
                    'ci_lower': val,    
                    'ci_upper': val    
                })    
            
        times = [stat['time'] for stat in time_stats]    
        means = [stat['mean'] for stat in time_stats]    
        ci_lower = [stat['ci_lower'] for stat in time_stats]    
        ci_upper = [stat['ci_upper'] for stat in time_stats]    
            
        color = COLORS.get(model_type, '#333333')    
        plt.plot(times, means, color=color, label=model_type, linewidth=2.5)    
        plt.fill_between(times, ci_lower, ci_upper, color=color, alpha=0.2)    
    
    plt.xlabel('Simulation Time (s)', fontweight='bold')    
    plt.ylabel(ylabel, fontweight='bold')    
    plt.title(f'{title}\nwith 95% Confidence Intervals', fontweight='bold', pad=20)    
    plt.legend(loc='best')    
    plt.grid(True, alpha=0.3)    
    plt.xticks(np.arange(0, 541, 60))    
    plt.xlim(0, 540)    
        
    plt.tight_layout()    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')    
    plt.close()    
    print(f"✓ Plot saved: {save_path}")  

def main():  
    """Main function"""  
    print("=== TEMPORAL EVALUATION: CNN vs TRADITIONAL ===")  
    print(f"Simulating {NUM_SCENARIOS} scenarios of {SIMULATION_TIME} seconds each")  
    
    output_dir = create_output_directory()  
    
    print("\n1. Running traditional traffic lights...")  
    traditional_metrics = run_traditional_scenarios()  
    
    print("\n2. Running CNN model...")  
    ai_metrics = run_ai_scenarios()  
    
    print("\n3. Processing temporal data...")  
    all_metrics = traditional_metrics + ai_metrics  
    df = pd.DataFrame(all_metrics)  
    
    csv_path = f"{output_dir}/temporal_metrics_combined.csv"  
    df.to_csv(csv_path, index=False)  
    print(f"✓ Data saved: {csv_path}")  
    
    print("\n4. Generating temporal plots...")  
    
    plot_temporal_evolution_with_ci(  
        df,   
        'system_mean_speed',  
        'Average Speed (m/s)',  
        'Temporal Evolution of Average Speed',  
        f"{output_dir}/temporal_speed_cnn.pdf"  
    )  
    
    plot_temporal_evolution_with_ci(    
        df,    
        'system_mean_waiting_time',     
        'Average Waiting Time (s)',  
        'Temporal Evolution of Average Waiting Time',  
        f"{output_dir}/temporal_waiting_time_cnn.pdf"  
    )  
    
    print("\n5. Generating statistical report...")  
    
    summary_stats = df.groupby('model_type').agg({  
        'system_mean_speed': ['mean', 'std'],  
        'system_mean_waiting_time': ['mean', 'std']  
    }).round(4)  
    
    print("\n=== SUMMARY STATISTICS ===")  
    print(summary_stats)  
    
    report_path = f"{output_dir}/statistical_report.txt"  
    with open(report_path, 'w', encoding='utf-8') as f:  
        f.write("TEMPORAL EVALUATION REPORT: CNN vs TRADITIONAL\n")  
        f.write("=" * 60 + "\n\n")  
        f.write(f"Number of scenarios: {NUM_SCENARIOS}\n")  
        f.write(f"Simulation time: {SIMULATION_TIME} seconds\n")  
        f.write(f"Delta time: {DELTA_TIME} seconds\n\n")  
        f.write("SUMMARY STATISTICS:\n")  
        f.write(str(summary_stats))  
        f.write("\n\n")  
    
    print(f"✓ Report saved: {report_path}")  
    
    print("\n" + "=" * 60)  
    print("✓ Temporal evaluation completed successfully!")  
    print(f"✓ Results saved in: {output_dir}")  
    print(f"✓ Generated files:")  
    print(f"  - {csv_path}")  
    print(f"  - {output_dir}/temporal_speed_cnn.pdf")  
    print(f"  - {output_dir}/temporal_waiting_time_cnn.pdf")  
    print(f"  - {report_path}")  

if __name__ == "__main__":  
    main()  
