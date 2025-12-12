import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- 1. Параметри Завдання ---

LAYER_SIZES = [2, 4, 8, 8, 6, 6, 1]
FEATURE_SIGNIFICANCE = [1.0, 1.0]

INTERVAL_MIN = 0
INTERVAL_MAX = 5


def target_function(x, y):
    return (x - y) * np.sin(x + y)


# --- 2. Функції Нейронної Мережі ---

def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def calculate_total_params(layer_sizes):
    total_params = 0
    for i in range(len(layer_sizes) - 1):
        weights_count = layer_sizes[i] * layer_sizes[i + 1]
        biases_count = layer_sizes[i + 1]
        total_params += (weights_count + biases_count)
    return total_params


def predict(inputs, chromosome, layer_sizes):
    current_idx = 0
    current_input = inputs

    for i in range(len(layer_sizes) - 1):
        input_dim = layer_sizes[i]
        output_dim = layer_sizes[i + 1]

        w_size = input_dim * output_dim
        W = chromosome[current_idx: current_idx + w_size].reshape((input_dim, output_dim))
        current_idx += w_size

        b_size = output_dim
        b = chromosome[current_idx: current_idx + b_size]
        current_idx += b_size

        output = np.dot(current_input, W) + b

        if i < len(layer_sizes) - 2:
            current_input = relu(output)
        else:
            current_input = linear(output)

    return current_input


# --- 3. Ініціалізація Нгуєна-Відроу ---

def create_population_nguyen_widrow(pop_size, layer_sizes, significance, min_val, max_val):
    population = []

    input_neurons = layer_sizes[0]  # 2
    hidden_neurons = layer_sizes[1]  # 4

    beta = 0.7 * (hidden_neurons ** (1.0 / input_neurons))

    for _ in range(pop_size):
        chromosome = []

        # --- Етап 1: Перший шар (Input -> Hidden 1) ---

        # 1. Генеруємо випадкові ваги від -0.5 до 0.5
        weights_l1 = np.random.uniform(-0.5, 0.5, (input_neurons, hidden_neurons))

        # 2. Нормалізуємо ваги для кожного нейрона (по стовпчиках) і множимо на Beta
        for j in range(hidden_neurons):
            norm = np.linalg.norm(weights_l1[:, j])
            if norm > 0:
                weights_l1[:, j] = (beta * weights_l1[:, j]) / norm

        # 3. Множимо ваги кожного входу на його коефіцієнт значущості
        for k in range(input_neurons):
            weights_l1[k, :] *= significance[k]

        # 4. Ініціалізація зміщень (Biases)
        # Зміщення обираються так, щоб інтервали активації покривали діапазон [min_val, max_val]
        # Евристика: випадкові зміщення в межах [-beta, beta], масштабовані на вхідний діапазон
        biases_l1 = np.random.uniform(-beta, beta, hidden_neurons)

        # Додаємо в хромосому
        chromosome.extend(weights_l1.flatten())
        chromosome.extend(biases_l1)

        # --- Етап 2: Решта шарів ---

        current_idx_layer = 1  # Починаємо з переходу шар 1 -> шар 2

        while current_idx_layer < len(layer_sizes) - 1:
            n_in = layer_sizes[current_idx_layer]
            n_out = layer_sizes[current_idx_layer + 1]

            # Ініціалізація He (Kaiming) або просто мала випадкова для глибоких шарів
            limit = np.sqrt(6 / (n_in + n_out))
            w_rest = np.random.uniform(-limit, limit, n_in * n_out)
            b_rest = np.random.uniform(-limit, limit, n_out)

            chromosome.extend(w_rest)
            chromosome.extend(b_rest)

            current_idx_layer += 1

        population.append(np.array(chromosome))

    return np.array(population)


# --- 4. Функції Генетичного Алгоритму ---

def calculate_fitness(chromosome, layer_sizes, X_data, y_true):
    y_pred = predict(X_data, chromosome, layer_sizes)
    mse = np.mean((y_true - y_pred) ** 2)
    return 1.0 / (mse + 1e-6)


def tournament_selection(population, fitness_scores, k=3):
    indices = np.random.randint(0, len(population), k)
    tournament_fitness = [fitness_scores[i] for i in indices]
    return population[indices[np.argmax(tournament_fitness)]]


def crossover(parent1, parent2):
    if len(parent1) < 2: return parent1
    point = np.random.randint(1, len(parent1) - 1)
    return np.concatenate((parent1[:point], parent2[point:]))


def mutate(chromosome, mutation_rate, mutation_strength):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] += np.random.normal(0, mutation_strength)
    return chromosome


# --- 5. Головний Процес ---

print("Початок навчання (Модифікований метод Нгуєна-Відроу)...")

# Параметри ГА
POPULATION_SIZE = 100
NUM_GENERATIONS = 2000
ELITISM_COUNT = 2
MUTATION_RATE = 0.05
# Адаптивна мутація для кращого результату
MUTATION_STRENGTH_START = 0.1
MUTATION_STRENGTH_END = 0.01
TOURNAMENT_SIZE = 5

chromosome_length = calculate_total_params(LAYER_SIZES)
print(f"Архітектура: {LAYER_SIZES}")

# Дані
NUM_SAMPLES = 200
X_train = np.random.uniform(INTERVAL_MIN, INTERVAL_MAX, (NUM_SAMPLES, 2))
y_train = target_function(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)


population = create_population_nguyen_widrow(
    POPULATION_SIZE,
    LAYER_SIZES,
    FEATURE_SIGNIFICANCE,
    INTERVAL_MIN,
    INTERVAL_MAX
)
# -------------------------

best_mse_overall = float('inf')
best_chromosome_overall = None

for gen in range(NUM_GENERATIONS):
    fitness_scores = [calculate_fitness(ind, LAYER_SIZES, X_train, y_train) for ind in population]

    sorted_indices = np.argsort(fitness_scores)[::-1]
    best_fitness = fitness_scores[sorted_indices[0]]
    best_mse = (1.0 / best_fitness) - 1e-6

    if best_mse < best_mse_overall:
        best_mse_overall = best_mse
        best_chromosome_overall = population[sorted_indices[0]].copy()

    if (gen % 100) == 0:
        print(f"Покоління {gen:4d} | MSE: {best_mse:.6f}")

    # Розрахунок адаптивної сили мутації
    current_mut_strength = MUTATION_STRENGTH_START - (gen / NUM_GENERATIONS) * (
                MUTATION_STRENGTH_START - MUTATION_STRENGTH_END)

    next_generation = []
    for i in range(ELITISM_COUNT):
        next_generation.append(population[sorted_indices[i]])

    while len(next_generation) < POPULATION_SIZE:
        parent1 = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)
        offspring = crossover(parent1, parent2)
        offspring = mutate(offspring, MUTATION_RATE, current_mut_strength)
        next_generation.append(offspring)

    population = np.array(next_generation)

print("Навчання завершено.")

# Тест
X_test = np.random.uniform(INTERVAL_MIN, INTERVAL_MAX, (NUM_SAMPLES, 2))
y_test_true = target_function(X_test[:, 0], X_test[:, 1]).reshape(-1, 1)
y_test_pred = predict(X_test, best_chromosome_overall, LAYER_SIZES)
test_mse = np.mean((y_test_true - y_test_pred) ** 2)
print(f"Підсумкова MSE: {test_mse:.6f}")

# Візуалізація
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
x_vals = np.linspace(INTERVAL_MIN, INTERVAL_MAX, 30)
y_vals = np.linspace(INTERVAL_MIN, INTERVAL_MAX, 30)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_true = target_function(X_grid, Y_grid)
ax1.plot_surface(X_grid, Y_grid, Z_true, cmap='viridis')
ax1.set_title("Оригінал")

ax2 = fig.add_subplot(122, projection='3d')
grid_inputs = np.stack((X_grid.ravel(), Y_grid.ravel()), axis=-1)
Z_pred = predict(grid_inputs, best_chromosome_overall, LAYER_SIZES).reshape(X_grid.shape)
ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap='plasma')
ax2.set_title(f"Нгуєн-Відроу Init\nMSE: {test_mse:.6f}")

plt.tight_layout()
plt.show()