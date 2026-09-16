import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuración de estilo para los gráficos
sns.set_theme(style="whitegrid")

print("=== INICIANDO PROYECTO DE CIENCIA DE DATOS ===")

# ==========================================
# ETAPA 1: Fundamentos y Preparación de Datos
# ==========================================
print("\n--- ETAPA 1: CREACIÓN Y LIMPIEZA DE DATOS ---")

# Creamos un dataset realista de rendimiento y estudio de manera local
# para garantizar que funcione al 100% en tu entorno.
np.random.seed(42)
n_filas = 150

horas_estudio = np.random.randint(1, 10, size=n_filas)
asistencia = np.random.randint(50, 100, size=n_filas)
# Simulamos una calificación basada en horas y asistencia con algo de ruido
calificacion = (horas_estudio * 4) + (asistencia * 0.4) + np.random.normal(0, 3, size=n_filas)

df = pd.DataFrame({
    'Horas_Estudio': horas_estudio,
    'Asistencia_Porcentaje': asistencia,
    'Calificacion_Final': calificacion
})

# Agregamos intencionalmente algunos valores duplicados y nulos para demostrar la limpieza
df.loc[0] = df.loc[1] # Duplicado intencional
df.loc[5, 'Calificacion_Final'] = np.nan # Nulo intencional

print("Dimensiones iniciales del dataset:", df.shape)
print("Valores nulos detectados antes de limpiar:\n", df.isnull().sum())

# Limpieza: eliminar duplicados y rellenar nulos numéricos con la mediana
df = df.drop_duplicates()
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Guardar el dataset limpio en la carpeta 'Data'
df.to_csv('Data/dataset_limpio.csv', index=False)
print("¡Limpieza exitosa! Dataset depurado guardado en la carpeta Data/dataset_limpio.csv")


# ==========================================
# ETAPA 2: Análisis Estadístico
# ==========================================
print("\n--- ETAPA 2: ANÁLISIS ESTADÍSTICO ---")
print(df.describe())

col_1 = 'Horas_Estudio'
col_2 = 'Calificacion_Final'

print(f"\nEstadísticas clave para '{col_1}':")
print(f" - Media: {df[col_1].mean():.2f}")
print(f" - Mediana: {df[col_1].median():.2f}")
print(f" - Desviación Estándar: {df[col_1].std():.2f}")

# Gráfico 1: Boxplot para identificar Outliers
plt.figure(figsize=(7, 4))
sns.boxplot(x=df[col_1], color="skyblue")
plt.title(f"Identificación de Outliers en {col_1}")
plt.xlabel(col_1)
plt.show()


# ==========================================
# ETAPA 3: Visualización y Modelo Predictivo
# ==========================================
print("\n--- ETAPA 3: VISUALIZACIÓN Y MODELO PREDICTIVO ---")

# Gráfico 2: Diagrama de dispersión (Scatterplot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x=col_1, y=col_2, alpha=0.7, color="purple")
plt.title(f"Relación Visual entre {col_1} y {col_2}")
plt.xlabel("Horas de Estudio")
plt.ylabel("Calificación Final")
plt.show()

# Modelo Predictivo Simple: Regresión Lineal
X = df[[col_1]] # Variable independiente
y = df[col_2]   # Variable a predecir

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
score_r2 = r2_score(y_test, y_pred)

print(f"Coeficiente de Pendiente del Modelo: {modelo.coef_[0]:.4f}")
print(f"Puntaje de Precisión del Modelo (R²): {score_r2:.4f}")
print("\n=== ¡PROYECTO EJECUTADO CON ÉXITO! ===")