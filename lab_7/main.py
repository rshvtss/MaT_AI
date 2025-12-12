import random
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Налаштування ГА ---
POPULATION_SIZE = 20
GENERATIONS = 200
MUTATION_RATE = 0.1
BOUNDS = (-10, 10)


# --- Функції ---
def func_1(x):
    val = x[0]
    return np.cos(np.sin(val)) * np.sin(val / 2)


def func_2(ind):
    x, y = ind[0], ind[1]
    # Мінімізуємо мінус функцію, щоб знайти максимум
    return -(x * np.sin(np.abs(y)))


def real_func_2(x, y):
    # Оригінальна функція для відображення
    return x * np.sin(np.abs(y))


# --- Клас ГА ---
class SimpleGA:
    def __init__(self, objective_func, n_vars, bounds, pop_size, generations, mutation_rate):
        self.func = objective_func
        self.n_vars = n_vars
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_history = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = [random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.n_vars)]
            self.population.append(ind)

    def get_fitness(self, individual):
        return self.func(individual)

    def selection(self):
        tournament = random.sample(self.population, k=3)
        tournament.sort(key=lambda ind: self.get_fitness(ind))
        return tournament[0]

    def crossover(self, parent1, parent2):
        child = []
        alpha = random.random()
        for i in range(self.n_vars):
            gene = alpha * parent1[i] + (1 - alpha) * parent2[i]
            child.append(gene)
        return child

    def mutate(self, individual):
        for i in range(self.n_vars):
            if random.random() < self.mutation_rate:
                noise = random.uniform(-0.5, 0.5)
                individual[i] += noise
                individual[i] = max(self.bounds[0], min(individual[i], self.bounds[1]))
        return individual

    def run(self):
        self.initialize_population()
        for _ in range(self.generations):
            self.population.sort(key=lambda ind: self.get_fitness(ind))
            # Зберігаємо найкращий результат поточного покоління
            self.best_history.append(self.get_fitness(self.population[0]))

            new_pop = [self.population[0]]  # Елітизм
            while len(new_pop) < self.pop_size:
                p1 = self.selection()
                p2 = self.selection()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            self.population = new_pop
        return self.population[0], self.best_history


# --- Запуск алгоритмів ---

# 1. Мінімізація
ga1 = SimpleGA(func_1, 1, BOUNDS, POPULATION_SIZE, GENERATIONS, MUTATION_RATE)
best_1, hist_1 = ga1.run()
y_min = func_1(best_1)
print(f"Завдання 1: Мінімум x = {best_1[0]:.4f}, y = {y_min:.4f}")

# 2. Максимізація
ga2 = SimpleGA(func_2, 2, BOUNDS, POPULATION_SIZE, GENERATIONS, MUTATION_RATE)
best_2, hist_2 = ga2.run()
z_max = real_func_2(best_2[0], best_2[1])
# Перетворюємо історію назад у позитивні числа для графіка максимізації
hist_2_positive = [-val for val in hist_2]
print(f"Завдання 2: Максимум x = {best_2[0]:.4f}, y = {best_2[1]:.4f}, z = {z_max:.4f}")

# --- Побудова графіків (Сітка 2x2) ---
fig = plt.figure(figsize=(12, 10))

# --- Ряд 1: Завдання 1 ---

# Графік 1.1: Функція
ax1 = fig.add_subplot(2, 2, 1)
x_vals = np.linspace(BOUNDS[0], BOUNDS[1], 400)
y_vals = np.cos(np.sin(x_vals)) * np.sin(x_vals / 2)
ax1.plot(x_vals, y_vals, label='Функція', color='blue')
ax1.scatter(best_1[0], y_min, color='red', s=80, label='Знайдений мінімум', zorder=5)
ax1.set_title('Завдання 1: Пошук мінімуму')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

# Графік 1.2: Збіжність
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(hist_1, color='green', marker='o', markersize=3)
ax2.set_title('Завдання 1: Графік збіжності')
ax2.set_xlabel('Покоління')
ax2.set_ylabel('Значення функції (Min)')
ax2.grid(True)

# --- Ряд 2: Завдання 2 ---

# Графік 2.1: 3D Поверхня
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
x_grid = np.linspace(BOUNDS[0], BOUNDS[1], 40)
y_grid = np.linspace(BOUNDS[0], BOUNDS[1], 40)
X, Y = np.meshgrid(x_grid, y_grid)
Z = X * np.sin(np.abs(Y))
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax3.scatter(best_2[0], best_2[1], z_max, color='red', s=100, label='Максимум')
ax3.set_title('Завдання 2: Пошук максимуму')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.legend()

# Графік 2.2: Збіжність
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(hist_2_positive, color='purple', marker='o', markersize=3)
ax4.set_title('Завдання 2: Графік збіжності')
ax4.set_xlabel('Покоління')
ax4.set_ylabel('Значення функції (Max)')
ax4.grid(True)

plt.tight_layout()
plt.show()