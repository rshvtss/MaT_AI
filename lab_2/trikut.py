import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error

x_values = np.linspace(0,20,100)
y_values = np.cos(np.sin(x_values)) * np.sin(x_values / 2)
z_values = x_values * np.sin(np.abs(y_values))

plt.plot(x_values, y_values)
plt.title("Y-function")
plt.show()

plt.plot(x_values, z_values)
plt.title("Z-function")
plt.show()

x_means = np.linspace(min(x_values), max(x_values), 6)
y_means = np.linspace(min(y_values), max(y_values), 6)
z_means = np.linspace(min(z_values), max(z_values), 9)

mx = [fuzz.trimf(x_values, [x_means[i] - 3, x_means[i], x_means[i] + 3]) for i in range(6)]
my =[fuzz.trimf(np.linspace(min(y_values), max(y_values), 100), [y_means[i] - 3, y_means[i], y_means[i] + 3])
    for i in range(6)]
mz =[fuzz.trimf(np.linspace(min(z_values), max(z_values), 100), [z_means[i] - 4, z_means[i], z_means[i] + 4])
    for i in range(9)]

for i in range(6):
    plt.plot(x_values, mx[i])
plt.title("X Trimf")
plt.show()

for i in range(6):
    plt.plot(np.linspace(min(y_values), max(y_values), 100), my[i])
plt.title("Y Trimf")
plt.show()

for i in range(9):
    plt.plot(np.linspace(min(z_values), max(z_values), 100), mz[i])
plt.title("Z Trimf")
plt.show()

def calculate_trimf(x, a, b, c):
    if a<= x < b:
        return (x-a)/(b-a)
    elif b <= x <= c:
        return (c-x)/(c-b)
    else:
        return 0


def functionCompare(value, means, diff):
    best_func_value = -float("inf")
    best_index = -1
    for index, mean in enumerate(means):
        ff = calculate_trimf(value, mean - diff, mean, mean + diff)
        if ff > best_func_value:
            best_func_value = ff
            best_index = index
    return best_index

print("Таблиця значень")
table = [["y\\x"] + [str(x) for x in x_means]]
for y_value in y_means:
    row = [round(y_value, 2)]
    for x in x_means:
        z = np.cos(y_value+x/2)
        row.append(round(z, 2))
    table.append(row)
print(tabulate(table, tablefmt="grid"))


rules = {}
print("Таблиця з назвами ф-цій")
table = [["y\\x"] + ["mx" + str(i) for i in range(1, 7)]]
for i in range(6):
    row = ["my" + str(i+1)]
    for j in range(6):
        z = x_means[j] * np.sin(np.abs(y_means[i]))
        best_func = functionCompare(z, z_means, 4)
        row.append("mf" + str(best_func + 1))
        rules[(j, i)] = best_func
    table.append(row)
print(tabulate(table, tablefmt = "grid"))

print("\nRules:")
for rule in rules.keys():
    print(f"if (x is mx{rule[0] + 1}) and (y is my{rule[1] + 1}) then (z is mf{rules[rule] + 1})")


z_output = []
for x in x_values:
    best_x_func = functionCompare(x, x_means, 3)
    best_y_func = functionCompare(np.cos(np.sin(x)) * np.sin(x / 2), y_means, 0.5)
    best_z_func = rules[(best_x_func, best_y_func)]
    z_output.append(z_means[best_z_func])

plt.plot(x_values, z_output, label="Model")
plt.plot(x_values, z_values, label="True")
plt.title("Справжня і змодельована ф-ції")
plt.legend()
plt.show()

mse = mean_squared_error(z_values, z_output)
mae = mean_absolute_error(z_values, z_output)
print(f"\nMean Squared Error (MSE) = {mse}")
print(f"\nMean Absolute Error (MAE) = {mae}")