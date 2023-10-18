import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Leer datos desde un archivo CSV
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Leer la primera fila como encabezado
        data = [row for row in reader]
    return header, data

# Función para normalizar características
def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized_features = (features - mean) / std
    return normalized_features, mean, std

# Función para calcular el error cuadrático medio
def mean_squared_error(predictions, target):
    return np.mean((predictions - target) ** 2)

# Función para realizar regresión polinómica múltiple con descenso de gradiente
def polynomial_regression_gradient_descent(features, target, degree, learning_rate=0.01, epochs=100000):
    # Agregar una columna de unos para el término independiente
    X = np.column_stack([np.ones(features.shape[0])] + [features ** i for i in range(1, degree + 1)])

    # Inicializar coeficientes aleatoriamente
    coefficients = np.random.rand(degree + 1)

    # Descenso de gradiente
    for epoch in range(epochs):
        predictions = X @ coefficients
        errors = predictions - target
        gradients = 2 * X.T @ errors / len(target)
        coefficients -= learning_rate * gradients

    return coefficients

# Función para graficar resultados
def plot_results(x, y, degree, coefficients):
    # Crear un conjunto de puntos para graficar la línea de regresión
    x_values = np.linspace(min(x), max(x), 100)
    y_values = np.polyval(coefficients[::-1], x_values)  # Calcula y para cada x usando los coeficientes

    # Graficar los puntos de datos
    plt.scatter(x, y, label='Puntos')

    # Graficar la línea de regresión
    plt.plot(x_values, y_values, label=f'Regression polinomial multiple (grado {degree})', color='red')

    # Configuración adicional del gráfico
    plt.title('Regresión Polinomial Multiple\nError cuadratico medio: ' + str(mse_test))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Datos del archivo CSV
csv_file_path = 'Fish.csv'
header, data = read_csv(csv_file_path)
data = np.array(data)

# Obtener las características (columnas 0 y 6) y la variable objetivo
species_column = data[:, 0]
width_column = data[:, 6].astype(float)
features = np.column_stack([species_column, width_column])
target = data[:, 1].astype(float)  # Tomar la columna de Weight como variable objetivo

# Label Encoding para la variable categórica "Species"
unique_species = np.unique(species_column)
species_mapping = {species: i for i, species in enumerate(unique_species)}
encoded_species = np.array([species_mapping[specie] for specie in species_column])

# Concatenar las características codificadas con las originales
features_encoded = np.column_stack((encoded_species, width_column))

# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
split_ratio = 0.3
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=split_ratio, random_state=42)

# Normalizar las características de entrenamiento y prueba
X_train_normalized, mean_train, std_train = normalize_features(X_train[:, 1:])
X_test_normalized = (X_test[:, 1] - mean_train) / std_train

# Realizar regresión polinómica múltiple con descenso de gradiente
degree = 5
coefficients = polynomial_regression_gradient_descent(X_train_normalized, y_train, degree)

# Imprimir el error cuadrático medio en el conjunto de prueba
X_test_poly = np.column_stack([np.ones(X_test_normalized.shape[0])] + [X_test_normalized ** i for i in range(1, degree + 1)])
predictions_test = X_test_poly @ coefficients
mse_test = mean_squared_error(predictions_test, y_test)
print(f'Mean Squared Error on Test Set: {mse_test}')

# Graficar resultados en el conjunto de prueba
plot_results(X_test_normalized, y_test, degree, coefficients)