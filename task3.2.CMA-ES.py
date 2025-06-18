import numpy as np
import pandas as pd
import random
import gymnasium as gym
from evogym.envs import *    
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import utils
import cma
import torch
import os
import csv
from datetime import datetime
import time
import multiprocessing as mp
from functools import partial

NUM_GENERATIONS = 100
POP_SIZE = 50
SIGMA_INIT = 0.2
STEPS = 500
SCENARIOS = ['ObstacleTraverser-v0']
SEEDS = [51, 52, 53, 54, 55] 

# Parelalização
NUM_PROCESSES = min(mp.cpu_count(), POP_SIZE)  

robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connectivity = get_full_connectivity(robot_structure)

# --- Funcões auxiliares para os pesos ---
def flatten_weights(weights_list):
    """Transforma uma lista de arrays de pesos em um vetor unidimensional"""
    return np.concatenate([w.flatten() for w in weights_list])

def structure_weights(flat_weights, model):
    """Transforma um vetor unidimensional em uma lista de arrays com as formas originais"""
    structured_weights = []
    current_idx = 0
    
    for param in model.parameters():
        shape = param.shape
        param_size = np.prod(shape)
        param_weights = flat_weights[current_idx:current_idx + param_size]
        structured_weights.append(param_weights.reshape(shape))
        current_idx += param_size
        
    return structured_weights

# --- Função para salvar dados de cada geração ---
def save_generation_data(generation, population, fitness_scores, scenario, controller_name, seed, parameters):
    """
    Salva os dados de cada geração em um arquivo CSV
    """
    folder = f"results_seed_{seed}/{controller_name}_{scenario}_CMA-ES"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Cabeçalho com metadados da run (apenas na primeira geração)
        if generation == 0:
            writer.writerow(["# ALGORITHM", parameters.get("algorithm", "CMA-ES")])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        # Cabeçalho dos dados por indivíduo
        writer.writerow(["Index", "Fitness", "Reward", "Weights"])

        for i, (weights, fitness) in enumerate(zip(population, fitness_scores)):
            weights_str = str(weights)  # Converter pesos para string para armazenar no CSV
            writer.writerow([i, -fitness, fitness, weights_str])
 
# --- Função para salvar resultados num Excel ---           
def save_results_to_excel(controller, best_fitness, scenario, population_size, num_generations, execution_time, seed, controller_weights, filename='task3_2_Results_Complete.xlsx'):
    """
    Salva os resultados em um arquivo Excel, incluindo os pesos e bias do controlador
    """
    # Converter os pesos e bias para uma string 
    weights_str = str(controller_weights)

    new_data = {
        'Scenario': [scenario],
        'Controller': [controller.__name__],
        'Population Size': [population_size],
        'Number of Generations': [num_generations],
        'Best Fitness': [best_fitness],
        'Execution Time (s)': [execution_time],
        'Seed': [seed],
        'Algorithm': ["CMA-ES"],
        'Controller Weights': [weights_str] 
    }

    new_df = pd.DataFrame(new_data)

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_excel(filename, index=False)
    print(f"Resultados salvos em {filename}")


# --- Função de Fitness ---
def evaluate_fitness(weights, scenario, input_size, output_size, view=False):
    """Função para avaliar o fitness de um indivíduo"""
    # Criamos um novo controlador para cada avaliação para garantir independência entre processos
    brain = NeuralController(input_size, output_size)
    set_weights(brain, weights)  # Load weights into the network
    
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env
    
    # Criar viewer apenas se necessário para visualização
    if view:
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
    
    state = env.reset()[0]  # Get initial state
    t_reward = 0
    
    for t in range(STEPS):  
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten() # Get action
        
        if view:
            viewer.render('screen') 
            
        state, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        
        if terminated or truncated:
            break
    
    if view:
        viewer.close()
    
    env.close()
    return t_reward 

# --- Função para avaliação paralela da fitness ---
def evaluate_population_parallel(population, scenario, input_size, output_size):
    """Avalia toda a população em paralelo"""
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        # Cria uma função parcial com os parâmetros fixos
        eval_func = partial(evaluate_fitness, scenario=scenario, input_size=input_size, output_size=output_size, view=False)
        # Avalia todos os indivíduos em paralelo
        fitness_results = pool.map(eval_func, population)
    
    return fitness_results

# ----- Funções para a evolução do controlador -----

def initialize_controller_population(input_size, output_size, size=POP_SIZE):
    """Inicializa a população de controladores usando CMA-ES"""
    # Criamos um modelo base para obter as dimensões dos pesos
    base_model = NeuralController(input_size, output_size)
    
    # Obtemos o número total de parâmetros para o CMA-ES
    flat_weights = flatten_weights(get_weights(base_model))
    cma_es = cma.CMAEvolutionStrategy(
        flat_weights,  # Inicialização com zeros
        SIGMA_INIT,  # Desvio padrão inicial
        {'popsize': size, "seed":42}
    )
    
    # Gera a população inicial de controladores
    cma_solutions = cma_es.ask()
    controller_population = []
    
    for solution in cma_solutions:
        structured_weights = structure_weights(solution, base_model)
        controller_population.append(structured_weights)
        
    return controller_population, cma_es, base_model

def evolve_controllers(controller_population, cma_es, base_model, fitness_values):
    """Evolui a população de controladores usando CMA-ES"""
    # Substituir valores não finitos por um valor padrão
    fitness_values = np.array(fitness_values)
    fitness_values[~np.isfinite(fitness_values)] = -100.0

    # Passa os dados de fitness para o CMA-ES (CMA-ES minimiza, então invertemos)
    cma_es.tell([flatten_weights(controller) for controller in controller_population], 
                [-fitness for fitness in fitness_values])
    
    # Gera a nova população
    cma_solutions = cma_es.ask()
    new_controllers = []
    
    for solution in cma_solutions:
        structured_weights = structure_weights(solution, base_model)
        new_controllers.append(structured_weights)
        
    return new_controllers

# ----- Funções de inicialização do ambiente -----
def get_input_output_sizes(scenario):
    """Obtém as dimensões de entrada/saída do ambiente"""
    global robot_structure
    
    connectivity = get_full_connectivity(robot_structure)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    
    # Obter dimensões de entrada e saída
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    
    env.close()
    
    print(f"Input Size: {input_size}, Output Size: {output_size}")
    
    return input_size, output_size

# --- Função principal para executar o algoritmo ---
def run_cma(seed, scenario):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Executando CMA-ES em paralelo com {NUM_PROCESSES} processos")

    input_size, output_size = get_input_output_sizes(scenario)

    # Inicializa a população de controladores
    controller_population, cma_es, base_model = initialize_controller_population(input_size, output_size)

    best_global_fitness = float('-inf')
    best_global_solution = None
    start = time.time()

    for generation in range(NUM_GENERATIONS):
        
        fitnesses = evaluate_population_parallel(controller_population, scenario, input_size, output_size)

        # Atualiza melhor fitness global
        best_gen_idx = np.argmax(fitnesses)
        if fitnesses[best_gen_idx] > best_global_fitness:
            best_global_fitness = fitnesses[best_gen_idx]
            best_global_solution = flatten_weights(controller_population[best_gen_idx])
            print(f"Novo melhor fitness: {best_global_fitness:.2f}")

        controller_population = evolve_controllers(controller_population, cma_es, base_model, fitnesses)

        # Salvar dados da geração
        parameters = {
            "algorithm": "CMA-ES",
            "population_size": POP_SIZE,
            "num_generations": NUM_GENERATIONS,
            "scenario": scenario,
            "steps": STEPS,
            "seed": seed,
            "controller_name": "NeuralController",
            "num_processes": NUM_PROCESSES
        }

        save_generation_data(
            generation=generation,
            population=controller_population,
            fitness_scores=fitnesses,
            scenario=scenario,
            controller_name="NeuralController",
            seed=seed,
            parameters=parameters
        )

        print(f"Geração {generation + 1}/{NUM_GENERATIONS} | Best fitness: {best_global_fitness:.2f}")

    #Aplica os melhores pesos encontrados no modelo base
    best_weights = structure_weights(best_global_solution, base_model)
    set_weights(base_model, best_weights)

    execution_time = time.time() - start
    print(f"Tempo total: {execution_time:.2f} segundos")
    print(f"Best fitness: {best_global_fitness:.2f}")

    controller_weights = flatten_weights(get_weights(base_model))

    # Salvar resultados num Excel
    save_results_to_excel(
        controller=NeuralController,
        best_fitness=best_global_fitness,
        scenario=scenario,
        population_size=POP_SIZE,
        num_generations=NUM_GENERATIONS,
        execution_time=execution_time,
        seed=seed,
        controller_weights=controller_weights
    )

    for _ in range(3):
        visualize_policy(best_weights, scenario, base_model, input_size, output_size)

    utils.create_gif_to_task3_2(robot_structure, filename=f'CMA-ES_{scenario}_seed{seed}.gif', scenario=scenario, steps=STEPS, controller=base_model)


def visualize_policy(weights, scenario, brain, input_size, output_size):
    set_weights(brain, weights)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]

    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    viewer.close()
    env.close()
    

# --- Função principal para execução do script ---
def main():
    mp.set_start_method('spawn', force=True)
    
    mode = input("Selecione o modo (1=Run único, 2=5 execuções por cenário): ")

    if mode == '1':
        num = input("Escolha o cenário pelo número (1: DownStepper-v0 ou 2: ObstacleTraverser-v0): ")
        
        if num == '1':
            scenario = 'DownStepper-v0'
        elif num == '2':
            scenario = 'ObstacleTraverser-v0'
        else:
            print("Cenário inválido.")
            return
        
        seed = int(input("Seed a utilizar: "))
        run_cma(seed, scenario)

    elif mode == '2':
        for scenario in SCENARIOS:
            for seed in SEEDS:
                print(f"\n--- Executando CMA-ES para o cenário {scenario} com a seed {seed} ---")
                run_cma(seed, scenario)
    else:
        print("Modo inválido.")

if __name__ == '__main__':
    main()