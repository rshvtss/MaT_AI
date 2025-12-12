import random
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- КОНСТАНТИ ---
NUM_NODES = 20
NUM_SWITCHES = 4
MAX_PORTS = 8
PENALTY_COEF = 1000000
GENERATIONS = 100
ELITISM_COUNT = 2
FIXED_SEED = 42  # Константа для фіксації випадковості

# --- ГЕНЕРАЦІЯ УМОВ (ОДНАКОВА ДЛЯ ВСІХ) ---
# Генеруємо матрицю трафіку один раз "назавжди"
np.random.seed(FIXED_SEED)
TRAFFIC_MATRIX = np.random.randint(0, 50, size=(NUM_NODES, NUM_NODES))
np.fill_diagonal(TRAFFIC_MATRIX, 0)
TRAFFIC_MATRIX = (TRAFFIC_MATRIX + TRAFFIC_MATRIX.T) // 2


# --- ФУНКЦІЇ ---

def create_chromosome():
    return [random.randint(0, NUM_SWITCHES - 1) for _ in range(NUM_NODES)]


def calculate_fitness(chromosome):
    external_traffic = 0
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            if chromosome[i] != chromosome[j]:
                external_traffic += TRAFFIC_MATRIX[i][j]

    switch_counts = [0] * NUM_SWITCHES
    for sw in chromosome:
        switch_counts[sw] += 1

    penalty = 0
    for count in switch_counts:
        if count > MAX_PORTS:
            penalty += (count - MAX_PORTS) * PENALTY_COEF
    return external_traffic + penalty


def crossover(parent1, parent2):
    point = random.randint(1, NUM_NODES - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        node_idx = random.randint(0, NUM_NODES - 1)
        chromosome[node_idx] = random.randint(0, NUM_SWITCHES - 1)
    return chromosome


def tournament_selection(population, fitnesses):
    indices = random.sample(range(len(population)), 3)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# --- ЕКСПЕРИМЕНТ ЗІ СКИДАННЯМ SEED ---

def run_experiment(pop_size, mutation_rate, label):
    print(f"Запуск тесту: {label}...")

    # !!! ГОЛОВНА ЗМІНА !!!
    # Скидаємо генератор перед кожним запуску.
    # Тепер кожен тест стартує з однакової "точки удачі".
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)

    population = [create_chromosome() for _ in range(pop_size)]
    history = []

    for gen in range(GENERATIONS):
        fitnesses = [calculate_fitness(ind) for ind in population]
        best_gen_fitness = min(fitnesses)
        history.append(best_gen_fitness)

        sorted_indices = np.argsort(fitnesses)
        new_population = [population[i] for i in sorted_indices[:ELITISM_COUNT]]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, mutation_rate))

        population = new_population

    final_fitnesses = [calculate_fitness(ind) for ind in population]
    best_idx = np.argmin(final_fitnesses)
    return history, population[best_idx], final_fitnesses[best_idx]


# --- ВІЗУАЛІЗАЦІЯ ---

if __name__ == "__main__":

    # Параметри для порівняння
    scenarios = [
        {"pop": 20, "mut": 0.1, "label": "Pop 20"},
        {"pop": 50, "mut": 0.1, "label": "Pop 50"},
        {"pop": 100, "mut": 0.1, "label": "Pop 100"},
        {"pop": 200, "mut": 0.1, "label": "Pop 200"}
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)

    global_best_solution = None
    global_best_score = float('inf')
    winner_label = ""

    for sc in scenarios:
        hist, sol, score = run_experiment(sc["pop"], sc["mut"], sc["label"])
        ax1.plot(hist, linewidth=2, label=f'{sc["label"]} (Min: {score})')

        if score < global_best_score:
            global_best_score = score
            global_best_solution = sol
            winner_label = sc["label"]

    ax1.set_title('Порівняння збіжності (при однакових стартових умовах)', fontsize=12)
    ax1.set_xlabel('Покоління')
    ax1.set_ylabel('Fitness')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Стовпчики для найкращого рішення
    switch_counts = [0] * NUM_SWITCHES
    for sw in global_best_solution:
        switch_counts[sw] += 1

    switches = [f'SW {i}' for i in range(NUM_SWITCHES)]
    colors = ['#2ca02c' if c <= MAX_PORTS else '#d62728' for c in switch_counts]
    bars = ax2.bar(switches, switch_counts, color=colors, edgecolor='black', alpha=0.8)

    ax2.axhline(y=MAX_PORTS, color='red', linestyle='--', label=f'Limit ({MAX_PORTS})')
    ax2.set_title(f'Завантаженість (Найкраще рішення з "{winner_label}")', fontsize=12)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
    ax2.legend()

    plt.show()