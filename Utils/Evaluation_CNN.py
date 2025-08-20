import os    
import sys    
import numpy as np    
import pandas as pd    
import matplotlib.pyplot as plt    
from pathlib import Path    
import ray    
from ray.rllib.algorithms.algorithm import Algorithm    
from ray.tune.registry import register_env    
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv    
import supersuit as ss    
from scipy import stats    
    
# Ensure SUMO_HOME is set    
if "SUMO_HOME" in os.environ:    
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")    
    sys.path.append(tools)    
else:    
    sys.exit("Please declare the environment variable 'SUMO_HOME'")    
    
import sumo_rl    
    
# --- Configuration ---     
USE_GUI = False    
CHECKPOINT_PATH = "../Outputs/ray_results/PPO_CNN_0/PPO_PPO_77c16_00000_0_2025-08-20_19-32-54/checkpoint_000091--81"    
COMPARISON_DIR = "../Results/CNN"    
SIMULATION_TIME = 540  # 9 minutos  
DELTA_TIME = 5    
NUM_SCENARIOS = 10  # 10 archivos de rutas diferentes    
  
# Configuración de matplotlib (igual que el ejemplo)  
plt.rcParams.update({  
    'figure.figsize': (12, 8),  
    'font.size': 22,            
    'axes.titlesize': 24,       
    'axes.labelsize': 22,       
    'xtick.labelsize': 20,      
    'ytick.labelsize': 20,      
    'legend.fontsize': 21,      
    'lines.linewidth': 2,     
    'grid.alpha': 0.3,  
    'axes.grid': True,  
    'figure.autolayout': True  
})  
    
COLORS = {    
    'traditional': '#d62728',    
    'ai': '#1f77b4',    
}    
  
def create_comparison_directory():    
    """Crea el directorio de comparación"""    
    os.makedirs(COMPARISON_DIR, exist_ok=True)    
    return COMPARISON_DIR    
  
def create_traditional_env(route_file):    
    """Crea el entorno SUMO-RL con semáforos tradicionales"""    
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
        fixed_ts=True,    
        single_agent=False    
    )    
    return env    
  
def create_ai_env(route_file):    
    """Crea el entorno SUMO-RL para el modelo de IA"""    
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
    """Extrae métricas del paso actual con tiempo de simulación"""    
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
  
def run_traditional_evaluation_all_scenarios():    
    """Ejecuta evaluación tradicional para todos los escenarios"""    
    print("Evaluando semáforos tradicionales en todos los escenarios...")    
        
    all_scenarios_metrics = []    
        
    for scenario in range(1, NUM_SCENARIOS + 1):    
        route_file = f"../Data/Routes_Test/routes{scenario}.rou.xml"    
        print(f"  Escenario {scenario}: {route_file}")    
            
        env = create_traditional_env(route_file)    
        scenario_metrics = []    
            
        observations = env.reset()    
        done = {"__all__": False}    
            
        while not done["__all__"]:    
            observations, rewards, done, info = env.step({})    
                
            current_time = env.sim_step    
            step_metrics = collect_step_metrics(info, current_time)    
            if step_metrics:    
                step_metrics['model_type'] = 'Tradicional'    
                step_metrics['scenario'] = scenario    
                scenario_metrics.append(step_metrics)    
            
        env.close()    
        all_scenarios_metrics.extend(scenario_metrics)    
        print(f"    ✓ Escenario {scenario} completado: {len(scenario_metrics)} puntos")    
        
    return all_scenarios_metrics    
  
def run_ai_evaluation_all_scenarios():    
    """Ejecuta evaluación de IA para todos los escenarios"""    
    print("Evaluando modelo de IA en todos los escenarios...")    
        
    ray.init(ignore_reinit_error=True)    
        
    # Registrar el entorno con la misma configuración del entrenamiento  
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
            route_file="../Data/routes.rou.xml",  # Usar el archivo original como default    
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
        
    # Cargar el algoritmo    
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)    
              
    all_scenarios_metrics = []    
        
    for scenario in range(1, NUM_SCENARIOS + 1):    
        route_file = f"../Data/Routes_Test/routes{scenario}.rou.xml"    
        print(f"  Escenario {scenario}: {route_file}")    
            
        # Crear entorno específico para este escenario    
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
                step_metrics['model_type'] = 'IA (PPO)'    
                step_metrics['scenario'] = scenario    
                scenario_metrics.append(step_metrics)    
                
            done = all(terminations.values()) or all(truncations.values())    
            
        env.close()    
        all_scenarios_metrics.extend(scenario_metrics)    
        print(f"    ✓ Escenario {scenario} completado: {len(scenario_metrics)} puntos")    
        
    ray.shutdown()    
    return all_scenarios_metrics    
  
def generate_scenario_based_plots_with_ci(df_combined, output_dir):    
    """Genera gráficas basadas en promedios por escenario con intervalos de confianza del 95%"""    
        
    # Calcular promedios por escenario (IGUAL QUE EL EJEMPLO)  
    scenario_summaries = df_combined.groupby(['scenario', 'model_type']).agg({    
        'system_mean_waiting_time': 'mean',    
        'system_mean_speed': 'mean',     
        'system_total_stopped': 'mean',    
        'system_total_arrived': 'last'    
    }).reset_index()    
        
    # Separar por tipo de modelo    
    trad_scenarios = scenario_summaries[scenario_summaries['model_type'] == 'Tradicional']    
    ai_scenarios = scenario_summaries[scenario_summaries['model_type'] == 'IA (PPO)']    
        
    # Función para calcular intervalos de confianza del 95%    
    def calculate_ci_95(data):    
        n = len(data)    
        mean = np.mean(data)    
        std_err = stats.sem(data)    
        ci = std_err * stats.t.ppf(0.975, n-1)    
        return mean, ci    
        
    # Calcular estadísticas para cada métrica (SOLO 2 MÉTRICAS PRINCIPALES)  
    metrics = {    
        'waiting_time': {    
            'trad_values': trad_scenarios['system_mean_waiting_time'].values,    
            'ai_values': ai_scenarios['system_mean_waiting_time'].values,    
            'ylabel': 'Tiempo de Espera Promedio (s)',    
            'title': 'Tiempo de Espera Promedio'    
        },    
        'speed': {    
            'trad_values': trad_scenarios['system_mean_speed'].values,    
            'ai_values': ai_scenarios['system_mean_speed'].values,    
            'ylabel': 'Velocidad Promedio (m/s)',    
            'title': 'Velocidad Promedio'    
        }  
    }    
        
    # Generar gráficas individuales para cada métrica (2 BARRAS POR GRÁFICA)  
    for i, (metric_name, metric_data) in enumerate(metrics.items(), 1):    
        plt.figure(figsize=(12, 8))    
            
        # Calcular estadísticas    
        trad_mean, trad_ci = calculate_ci_95(metric_data['trad_values'])    
        ai_mean, ai_ci = calculate_ci_95(metric_data['ai_values'])    
            
        # Crear gráfica de barras con intervalos de confianza (EXACTAMENTE COMO EL EJEMPLO)  
        x_pos = [0, 1]    
        means = [trad_mean, ai_mean]    
        cis = [trad_ci, ai_ci]    
        colors = [COLORS['traditional'], COLORS['ai']]    
        labels = ['Semáforos Tradicionales', 'Modelo IA (PPO-CNN)']    
            
        bars = plt.bar(x_pos, means, color=colors, alpha=0.7, width=0.6)      
        plt.errorbar(x_pos, means, yerr=cis, fmt='none', color='black',       
                    capsize=10, capthick=2, elinewidth=2)  
          
        # Configurar etiquetas y título  
        plt.xticks(x_pos, labels)  
        plt.ylabel(metric_data['ylabel'])  
        plt.title(f"{metric_data['title']}\n(Promedio de {NUM_SCENARIOS} escenarios con IC 95%)",     
         fontsize=14, fontweight='bold', pad=30)  # Aumentar pad de 20 a 30
        #plt.title(metric_data['title'])  
        plt.grid(True, alpha=0.3)  
          
        # Añadir valores en las barras  
        for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):  
            height = bar.get_height()+bar.get_height()*0.015
            desplazamiento =  bar.get_width() * 0.25 # <-- Puedes ajustar este valor. Prueba con 0.05, 0.1, 0.2, etc.

            plt.text(
            (bar.get_x() + bar.get_width() / 2.) - desplazamiento,  # <-- Coordenada X modificada
            height,
            f'{mean:.2f} ± {ci:.2f}',
            ha='center',
            va='center',
            fontweight='bold',
            fontsize=14,
            color='black'
            )
            
            
          
        # Guardar gráfica  
        plt.tight_layout()  
        plt.savefig(f"{output_dir}/comparison_{metric_name}.png", dpi=300, bbox_inches='tight')  
        plt.savefig(f"{output_dir}/comparison_{metric_name}.pdf", bbox_inches='tight')  
        plt.show()  
        print(f"  ✓ Gráfica {metric_name} guardada")  
  
def perform_statistical_tests(df_combined):  
    """Realiza pruebas estadísticas entre los dos modelos"""  
    print("\nRealizando pruebas estadísticas...")  
      
    # Separar datos por tipo de modelo  
    trad_data = df_combined[df_combined['model_type'] == 'Tradicional']  
    ai_data = df_combined[df_combined['model_type'] == 'IA (PPO)']  
      
    # Calcular promedios por escenario para cada métrica  
    trad_scenarios = trad_data.groupby('scenario').agg({  
        'system_mean_waiting_time': 'mean',  
        'system_mean_speed': 'mean'  
    }).reset_index()  
      
    ai_scenarios = ai_data.groupby('scenario').agg({  
        'system_mean_waiting_time': 'mean',  
        'system_mean_speed': 'mean'  
    }).reset_index()  
      
    results = {}  
      
    # Prueba t-test para tiempo de espera  
    t_stat_wait, p_val_wait = stats.ttest_rel(  
        trad_scenarios['system_mean_waiting_time'],   
        ai_scenarios['system_mean_waiting_time']  
    )  
      
    # Prueba t-test para velocidad  
    t_stat_speed, p_val_speed = stats.ttest_rel(  
        trad_scenarios['system_mean_speed'],   
        ai_scenarios['system_mean_speed']  
    )  
      
    results['waiting_time'] = {  
        't_statistic': t_stat_wait,  
        'p_value': p_val_wait,  
        'significant': p_val_wait < 0.05  
    }  
      
    results['speed'] = {  
        't_statistic': t_stat_speed,  
        'p_value': p_val_speed,  
        'significant': p_val_speed < 0.05  
    }  
      
    # Mostrar resultados  
    print(f"Tiempo de espera - t-statistic: {t_stat_wait:.4f}, p-value: {p_val_wait:.4f}")  
    print(f"Velocidad - t-statistic: {t_stat_speed:.4f}, p-value: {p_val_speed:.4f}")  
      
    return results  
  
def save_summary_report(df_combined, statistical_results, output_dir):  
    """Guarda un reporte resumen de los resultados"""  
    print("\nGenerando reporte resumen...")  
      
    # Calcular estadísticas descriptivas  
    summary_stats = df_combined.groupby('model_type').agg({  
        'system_mean_waiting_time': ['mean', 'std', 'min', 'max'],  
        'system_mean_speed': ['mean', 'std', 'min', 'max'],  
        'system_total_arrived': ['mean', 'std']  
    }).round(4)  
      
    # Crear reporte  
    report_path = f"{output_dir}/summary_report.txt"  
    with open(report_path, 'w', encoding='utf-8') as f:  
        f.write("REPORTE DE COMPARACIÓN: SEMÁFOROS TRADICIONALES vs IA (PPO)\n")  
        f.write("=" * 60 + "\n\n")  
          
        f.write("CONFIGURACIÓN DEL EXPERIMENTO:\n")  
        f.write(f"- Número de escenarios: {NUM_SCENARIOS}\n")  
        f.write(f"- Tiempo de simulación: {SIMULATION_TIME} segundos\n")  
        f.write(f"- Delta time: {DELTA_TIME} segundos\n\n")  
          
        f.write("ESTADÍSTICAS DESCRIPTIVAS:\n")  
        f.write(str(summary_stats))  
        f.write("\n\n")  
          
        f.write("PRUEBAS ESTADÍSTICAS (t-test pareado):\n")  
        for metric, results in statistical_results.items():  
            f.write(f"{metric.upper()}:\n")  
            f.write(f"  t-statistic: {results['t_statistic']:.4f}\n")  
            f.write(f"  p-value: {results['p_value']:.4f}\n")  
            f.write(f"  Significativo (α=0.05): {'Sí' if results['significant'] else 'No'}\n\n")  
      
    print(f"  ✓ Reporte guardado en: {report_path}")  
  
def main():  
    """Función principal que ejecuta todo el análisis"""  
    print("Iniciando análisis estadístico comparativo...")  
    print("=" * 50)  
      
    # Crear directorio de resultados  
    output_dir = create_comparison_directory()  
      
    # Ejecutar evaluaciones  
    traditional_metrics = run_traditional_evaluation_all_scenarios()  
    ai_metrics = run_ai_evaluation_all_scenarios()  
      
    # Combinar datos  
    all_metrics = traditional_metrics + ai_metrics  
    df_combined = pd.DataFrame(all_metrics)  
      
    # Guardar datos combinados  
    df_combined.to_csv(f"{output_dir}/combined_metrics.csv", index=False)  
    print(f"\n✓ Datos combinados guardados: {len(all_metrics)} registros")  
      
    # Generar gráficas  
    print("\nGenerando gráficas comparativas...")  
    generate_scenario_based_plots_with_ci(df_combined, output_dir)  
      
    # Realizar pruebas estadísticas  
    statistical_results = perform_statistical_tests(df_combined)  
      
    # Generar reporte resumen  
    save_summary_report(df_combined, statistical_results, output_dir)  
      
    print("\n" + "=" * 50)  
    print("✓ Análisis completado exitosamente!")  
    print(f"✓ Resultados guardados en: {output_dir}")  
  
if __name__ == "__main__":  
    main()
                    
                    
                    