import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Функції membership ---
def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x < c:
        return (c - x) / (c - b)
    return 0.0

def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0

def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma**2))

# --- Параметри (відповідні до попередніх реалізацій) ---
x_values = np.linspace(0,20,100)
y_values = np.cos(np.sin(x_values)) * np.sin(x_values / 2)
z_values = x_values * np.sin(np.abs(y_values))

x_means = np.linspace(min(x_values), max(x_values), 6)
y_means = np.linspace(min(y_values), max(y_values), 6)
z_means = np.linspace(min(z_values), max(z_values), 9)

# functionCompare для кожного типу
def best_index_trimf(value, means, diff):
    best_mu = -1.0; best_idx = -1
    for i, m in enumerate(means):
        mu = trimf(value, m-diff, m, m+diff)
        if mu > best_mu:
            best_mu = mu; best_idx = i
    return best_idx

def best_index_trapmf(value, means, half_width):
    # використовуємо параметри як у прикладі: [m-2*half, m-half, m+half, m+2*half]
    best_mu = -1.0; best_idx = -1
    for i, m in enumerate(means):
        a = m - 2*half_width; b = m - half_width; c = m + half_width; d = m + 2*half_width
        mu = trapmf(value, a, b, c, d)
        if mu > best_mu:
            best_mu = mu; best_idx = i
    return best_idx

def best_index_gauss(value, means, sigma):
    best_mu = -1.0; best_idx = -1
    for i,m in enumerate(means):
        mu = gaussmf(value, m, sigma)
        if mu > best_mu:
            best_mu = mu; best_idx = i
    return best_idx

# --- Збір правил для кожного MF типу (залежно від того, як ти будував правила) ---
def build_rules(best_index_x_fn, best_index_y_fn, z_means, x_means, y_means):
    rules = {}
    for j in range(len(x_means)):
        for i in range(len(y_means)):
            z = x_means[j] * np.sin(np.abs(y_means[i]))
            # для побудови правила шукаємо найкращий індекс по z_means
            # використаємо ту ж функцію, але для простоти — використовуємо trimf-like логіку: diff_z=4
            # тобто знаходимо index в z_means найближчий за membership (тут просто argmax трьома варіантами)
            # щоб зберегти сумісність - просто обчислимо відстані та візьмемо найменшу відстань
            zi = np.argmin(np.abs(z_means - z))
            rules[(j,i)] = int(zi)
    return rules

# Будуємо правила (одні й ті ж) для всіх трьох pipeline'ів — це відповідає твоїй початковій логіці
rules = build_rules(None, None, z_means, x_means, y_means)

# --- Обчислюємо outputs для кожного MF типу ---
z_out_trim = []
z_out_trap = []
z_out_gauss = []

best_x_trim = []
best_x_trap = []
best_x_gauss = []

best_y_trim = []
best_y_trap = []
best_y_gauss = []

for x in x_values:
    # corresponding y (як у твоєму коді)
    y = np.cos(np.sin(x)) * np.sin(x / 2)

    bxt = best_index_trimf(x, x_means, 3)
    bxtp = best_index_trapmf(x, x_means, 2)   # half_width=2 відповідає [m-4,m-2,m+2,m+4]
    bxg = best_index_gauss(x, x_means, 2.0)

    byt = best_index_trimf(y, y_means, 0.5)
    bytp = best_index_trapmf(y, y_means, 0.5) # half_width=0.5 -> [m-1,m-0.5,m+0.5,m+1]
    byg = best_index_gauss(y, y_means, 0.4)

    best_x_trim.append(bxt); best_x_trap.append(bxtp); best_x_gauss.append(bxg)
    best_y_trim.append(byt); best_y_trap.append(bytp); best_y_gauss.append(byg)

    z_out_trim.append(z_means[rules[(bxt, byt)]])
    z_out_trap.append(z_means[rules[(bxtp, bytp)]])
    z_out_gauss.append(z_means[rules[(bxg, byg)]])

# --- Помилки ---
mse_trim = mean_squared_error(z_values, z_out_trim)
mae_trim = mean_absolute_error(z_values, z_out_trim)

mse_trap = mean_squared_error(z_values, z_out_trap)
mae_trap = mean_absolute_error(z_values, z_out_trap)

mse_gauss = mean_squared_error(z_values, z_out_gauss)
mae_gauss = mean_absolute_error(z_values, z_out_gauss)

print("Trimf: MSE=", mse_trim, " MAE=", mae_trim)
print("Trapmf: MSE=", mse_trap, " MAE=", mae_trap)
print("Gaussmf: MSE=", mse_gauss, " MAE=", mae_gauss)

# --- Де вони різняться? ---
diff_x_trim_trap = np.sum(np.array(best_x_trim) != np.array(best_x_trap))
diff_y_trim_trap = np.sum(np.array(best_y_trim) != np.array(best_y_trap))

diff_x_trim_gauss = np.sum(np.array(best_x_trim) != np.array(best_x_gauss))
diff_y_trim_gauss = np.sum(np.array(best_y_trim) != np.array(best_y_gauss))

print("\nРізні best_x індекси (trim vs trap):", diff_x_trim_trap, " (trim vs gauss):", diff_x_trim_gauss)
print("Різні best_y індекси (trim vs trap):", diff_y_trim_trap, " (trim vs gauss):", diff_y_trim_gauss)

# Вивід індексів x, де результати z_out різні між trim і trap (перші 20)
diff_positions = np.where(np.array(z_out_trim) != np.array(z_out_trap))[0]
print("\nПозиції, де z_out відрізняється (trim vs trap) [перші 20]:", diff_positions[:20])
