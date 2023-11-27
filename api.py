import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Reemplaza 'TU_CLAVE_API' con la clave API que obtienes al registrarte en OpenWeatherMap
api_key = '99f759fc112c9984ef054a3f06228d0e'
url = 'https://api.openweathermap.org/data/2.5/forecast'

# Parámetros de la solicitud para una ciudad específica (por ejemplo, Nueva York)
params = {'q': 'New York,US', 'appid': api_key, 'units': 'metric'}

# Realizar la solicitud GET a la API de OpenWeatherMap
response = requests.get(url, params=params)

# Verificar si la solicitud fue exitosa (código de estado 200)
if response.status_code == 200:
    # Convertir la respuesta a formato JSON
    data = response.json()

    # Extraer datos relevantes (temperatura y humedad) de las próximas 5 horas
    weather_data = pd.DataFrame({'Temperature': [entry['main']['temp'] for entry in data['list']],
                                 'Humidity': [entry['main']['humidity'] for entry in data['list']]})

    # Mostrar los datos
    print(weather_data)

    # Análisis de datos y visualización
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(weather_data['Temperature'], label='Temperatura (°C)', marker='o')
    plt.title('Evolución de la Temperatura')
    plt.xlabel('Hora')
    plt.ylabel('Temperatura (°C)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(weather_data['Humidity'], label='Humedad (%)', marker='o', color='orange')
    plt.title('Evolución de la Humedad')
    plt.xlabel('Hora')
    plt.ylabel('Humedad (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Machine Learning - Predicción de la humedad a partir de la temperatura
    X = weather_data[['Temperature']]
    y = weather_data['Humidity']

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Visualizar la regresión lineal
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='black', label='Real')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicción')
    plt.title('Regresión lineal - Predicción de Humedad basada en Temperatura')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Humedad (%)')
    plt.legend()
    plt.show()

    # Calcular el error cuadrático medio en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)
    print(f'Error cuadrático medio en el conjunto de prueba: {mse:.2f}')
else:
    print(f'Error en la solicitud: {response.status_code}')
