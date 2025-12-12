import random
import numpy as np
import matplotlib

# Налаштування бекенду, як ти просив
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- КОНСТАНТИ ТА НАЛАШТУВАННЯ ---
NUM_NODES = 20  # Кількість вузлів
NUM_SWITCHES = 4  # Кількість комутаторів
MAX_PORTS_PER_SWITCH = 8  # Ліміт портів
PENALTY_SCORE = 1000000  # Штраф

GENERATIONS = 100  # Кількість поколінь
MUTATION_RATE = 0.1  # Ймовірність мутації
ELITISM_COUNT = 2  # Кількість елітних хромосом

# --- ГЕНЕРАЦІЯ ВХІДНИХ ДАНИХ ---
# Генеруємо матрицю один раз, щоб умови були однакові для обох експериментів
np.random.seed(42)
TRAFFIC_MATRIX = np.random.randint(0, 100, size=(NUM_NODES, NUM_NODES))
np.fill_diagonal(TRAFFIC_MATRIX, 0)


# --- ГЕНЕТИЧНИЙ АЛГОРИТМ (Твої функції) ---

def create_chromosome():
    return [random.randint(0, NUM_SWITCHES - 1) for _ in range(NUM_NODES)]


def calculate_fitness(chromosome):
    backbone_traffic = 0
    # 1. Трафік
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            if chromosome[i] != chromosome[j]:
                backbone_traffic += TRAFFIC_MATRIX[i][j]
                backbone_traffic += TRAFFIC_MATRIX[j][i]

    # 2. Штрафи
    switch_counts = [0] * NUM_SWITCHES
    for switch_id in chromosome:
        switch_counts[switch_id] += 1

    penalty = 0
    for count in switch_counts:
        if count > MAX_PORTS_PER_SWITCH:
            penalty += (count - MAX_PORTS_PER_SWITCH) * PENALTY_SCORE

    return backbone_traffic + penalty


def crossover(parent1, parent2):
    point = random.randint(1, NUM_NODES - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        gene_idx = random.randint(0, NUM_NODES - 1)
        chromosome[gene_idx] = random.randint(0, NUM_SWITCHES - 1)
    return chromosome


def selection(population, fitnesses):
    # Турнір 3
    tournament = random.sample(list(zip(population, fitnesses)), 3)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]


# --- ФУНКЦІЯ ЗАПУСКУ ЕКСПЕРИМЕНТУ ---
def run_experiment(pop_size, experiment_name):
    print(f"\n--- Запуск: {experiment_name} (Популяція: {pop_size}) ---")

    population = [create_chromosome() for _ in range(pop_size)]
    history = []

    for generation in range(GENERATIONS):
        fitnesses = [calculate_fitness(chrom) for chrom in population]
        best_fit = min(fitnesses)
        history.append(best_fit)

        # Елітаризм
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
        new_population = [x[0] for x in sorted_pop[:ELITISM_COUNT]]

        while len(new_population) < pop_size:
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2))

        population = new_population

    print(f"Фінальний результат: {history[-1]}")
    return history


# --- ГОЛОВНИЙ БЛОК (ВІЗУАЛІЗАЦІЯ) ---
if __name__ == "__main__":
    # Проводимо два експерименти згідно зі звітом
    history_50 = run_experiment(50, "Експеримент 1")
    history_100 = run_experiment(100, "Експеримент 2")

    # Побудова графіків
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.3)

    # Графік 1 (Популяція 50)
    ax1.plot(history_50, color='#1f77b4', linewidth=2)
    ax1.set_title('Експеримент 1: Розмір популяції 50', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Покоління')
    ax1.set_ylabel('Fitness (Трафік + Штрафи)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    # Позначка мінімуму
    min_val_1 = min(history_50)
    ax1.annotate(f'Min: {min_val_1}',
                 xy=(history_50.index(min_val_1), min_val_1),
                 xytext=(10, min_val_1 + 500),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Графік 2 (Популяція 100)
    ax2.plot(history_100, color='#2ca02c', linewidth=2)
    ax2.set_title('Експеримент 2: Розмір популяції 100', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Покоління')
    ax2.set_ylabel('Fitness (Трафік + Штрафи)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    # Позначка мінімуму
    min_val_2 = min(history_100)
    ax2.annotate(f'Min: {min_val_2}',
                 xy=(history_100.index(min_val_2), min_val_2),
                 xytext=(10, min_val_2 + 500),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    print("\nГрафіки згенеровано. Перевірте вікно.")
    plt.show()