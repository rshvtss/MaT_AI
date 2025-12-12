import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- 1. Параметри Завдання ---

# Архітектура мережі: 2 входи, 5 прихованих шарів, 1 вихід
LAYER_SIZES = [2, 4, 8, 8, 6, 6, 1]


def target_function(x, y):
    """Цільова функція, яку моделюємо: z = (x-y)*sin(x+y)"""
    return (x - y) * np.sin(x + y)


# Інтервал навчання [MIN, MAX]
INTERVAL_MIN = 0
INTERVAL_MAX = 5


# --- 2. Функції Нейронної Мережі ---

def relu(x):
    """Активаційна функція ReLU для прихованих шарів"""
    return np.maximum(0, x)


def linear(x):
    """Лінійна активація для вихідного шару (регресія)"""
    return x


def calculate_total_params(layer_sizes):
    """
    Обчислює загальну кількість параметрів (ваг + зміщень) для хромосоми ГА.
    """
    total_params = 0
    for i in range(len(layer_sizes) - 1):
        # Ваги: input_size * output_size
        weights_count = layer_sizes[i] * layer_sizes[i + 1]
        # Зміщення: output_size
        biases_count = layer_sizes[i + 1]
        total_params += (weights_count + biases_count)
    return total_params


def predict(inputs, chromosome, layer_sizes):
    """
    Виконує пряме поширення, використовуючи ваги з "хромосоми".
    """
    current_idx = 0
    current_input = inputs

    for i in range(len(layer_sizes) - 1):
        input_dim = layer_sizes[i]
        output_dim = layer_sizes[i + 1]

        # 1. "Розгортаємо" ваги з хромосоми
        w_size = input_dim * output_dim
        W = chromosome[current_idx: current_idx + w_size].reshape((input_dim, output_dim))
        current_idx += w_size

        # 2. "Розгортаємо" зміщення з хромосоми
        b_size = output_dim
        b = chromosome[current_idx: current_idx + b_size]
        current_idx += b_size

        # 3. Обчислюємо вихід шару: z = xW + b
        output = np.dot(current_input, W) + b

        # 4. Застосовуємо активацію
        if i < len(layer_sizes) - 2:  # Якщо це прихований шар
            current_input = relu(output)
        else:  # Якщо це вихідний шар
            current_input = linear(output)

    return current_input


# --- 3. Функції Генетичного Алгоритму ---

def calculate_fitness(chromosome, layer_sizes, X_data, y_true):
    """
    Функція придатності. Чим нижча помилка (MSE), тим вища придатність.
    """
    y_pred = predict(X_data, chromosome, layer_sizes)
    mse = np.mean((y_true - y_pred) ** 2)

    # Додаємо 1e-6, щоб уникнути ділення на нуль, якщо MSE = 0
    fitness = 1.0 / (mse + 1e-6)
    return fitness


def tournament_selection(population, fitness_scores, k=3):
    """
    Турнірний відбір: обираємо k випадкових особин,
    і перемагає та, у якої придатність найвища.
    """
    indices = np.random.randint(0, len(population), k)
    tournament_fitness = [fitness_scores[i] for i in indices]
    winner_local_idx = np.argmax(tournament_fitness)
    return population[indices[winner_local_idx]]


def crossover(parent1, parent2):
    """
    Одноточкове схрещування.
    """
    if len(parent1) < 2: return parent1  # Запобіжник
    point = np.random.randint(1, len(parent1) - 1)
    offspring = np.concatenate((parent1[:point], parent2[point:]))
    return offspring


def mutate(chromosome, mutation_rate, mutation_strength):
    """
    Мутація: додаємо невеликий "шум" до деяких генів (ваг).
    """
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            # Додаємо випадкове значення з нормального розподілу
            chromosome[i] += np.random.normal(0, mutation_strength)
    return chromosome


# --- 4. Головний Процес Навчання ---

print("Початок навчання нейронної мережі генетичним алгоритмом...")

# Параметри ГА
POPULATION_SIZE = 100  # Кількість мереж у поколінні
NUM_GENERATIONS = 2000  # Кількість поколінь
ELITISM_COUNT = 2  # Кількість найкращих особин, що переходять у наступне покоління
MUTATION_RATE = 0.05  # Ймовірність мутації гена
MUTATION_STRENGTH = 0.1  # Сила мутації (наскільки змінюються ваги)
TOURNAMENT_SIZE = 5  # Розмір турніру для відбору

# 1. Визначаємо розмір хромосоми
chromosome_length = calculate_total_params(LAYER_SIZES)
print(f"Архітектура: {LAYER_SIZES}")
print(f"Загальна кількість параметрів (довжина хромосоми): {chromosome_length}")

# 2. Генеруємо навчальні дані
NUM_SAMPLES = 200
X_train = np.random.uniform(INTERVAL_MIN, INTERVAL_MAX, (NUM_SAMPLES, 2))
y_train = target_function(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)

# 3. Ініціалізуємо початкову популяцію
# (Ваги ініціалізуються випадково в діапазоні [-1, 1])
population = np.random.uniform(-1.0, 1.0, (POPULATION_SIZE, chromosome_length))

# 4. Головний цикл еволюції
best_mse_overall = float('inf')

for gen in range(NUM_GENERATIONS):
    # 4.1. Оцінка придатності популяції
    fitness_scores = [calculate_fitness(ind, LAYER_SIZES, X_train, y_train) for ind in population]

    # 4.2. Сортування (найкращі - перші)
    sorted_indices = np.argsort(fitness_scores)[::-1]
    best_fitness = fitness_scores[sorted_indices[0]]
    best_mse = (1.0 / best_fitness) - 1e-6

    if best_mse < best_mse_overall:
        best_mse_overall = best_mse

    if (gen % 100) == 0:
        print(f"Покоління {gen:4d} | Найкраща MSE: {best_mse:.6f} | (Загалом найкраща: {best_mse_overall:.6f})")

    # 4.3. Створення нового покоління
    next_generation = []

    # Елітизм
    for i in range(ELITISM_COUNT):
        next_generation.append(population[sorted_indices[i]])

    # Схрещування та мутація
    while len(next_generation) < POPULATION_SIZE:
        # Відбір
        parent1 = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)

        # Схрещування
        offspring = crossover(parent1, parent2)

        # Мутація
        offspring = mutate(offspring, MUTATION_RATE, MUTATION_STRENGTH)

        next_generation.append(offspring)

    population = np.array(next_generation)

print("Навчання завершено.")

# 5. Отримуємо найкращу мережу
final_fitness_scores = [calculate_fitness(ind, LAYER_SIZES, X_train, y_train) for ind in population]
best_chromosome = population[np.argmax(final_fitness_scores)]

# 6. Тестування на нових даних
print("Тестування на новому наборі даних...")
X_test = np.random.uniform(INTERVAL_MIN, INTERVAL_MAX, (NUM_SAMPLES, 2))
y_test_true = target_function(X_test[:, 0], X_test[:, 1]).reshape(-1, 1)
y_test_pred = predict(X_test, best_chromosome, LAYER_SIZES)
test_mse = np.mean((y_test_true - y_test_pred) ** 2)

print(f"Підсумкова MSE на тестових даних: {test_mse:.6f}")

# --- 5. Візуалізація Результату ---

print("Створення 3D-графіків для порівняння...")

fig = plt.figure(figsize=(16, 8))

# Графік 1: Оригінальна функція
ax1 = fig.add_subplot(121, projection='3d')
x_vals = np.linspace(INTERVAL_MIN, INTERVAL_MAX, 30)
y_vals = np.linspace(INTERVAL_MIN, INTERVAL_MAX, 30)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z_true = target_function(X_grid, Y_grid)

ax1.plot_surface(X_grid, Y_grid, Z_true, cmap='viridis')
ax1.set_title(f"Оригінальна функція:\nz = (x-y)*sin(x+y)", fontsize=14)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Графік 2: Функція, змодельована НМ
ax2 = fig.add_subplot(122, projection='3d')
# Готуємо дані для сітки
grid_inputs = np.stack((X_grid.ravel(), Y_grid.ravel()), axis=-1)
# Отримуємо прогнози для всієї сітки
Z_pred = predict(grid_inputs, best_chromosome, LAYER_SIZES)
# Повертаємо у форму сітки
Z_pred = Z_pred.reshape(X_grid.shape)

ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap='plasma')
ax2.set_title(f"Змодельована НМ (Архітектура {LAYER_SIZES})\nTest MSE: {test_mse:.6f}", fontsize=14)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z (predicted)")

plt.tight_layout()
plt.show()