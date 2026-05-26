
# 🚢 Titanic · MLP Classifier

**TP5 — Inteligencia Artificial · RNA No Simbólica**  
Clasificación binaria de supervivencia en el Titanic mediante Redes Neuronales Artificiales del tipo Multilayer Perceptron (MLP).

---

## 📋 Descripción

Aplicación interactiva desarrollada en Python con Streamlit que entrena y compara 3 modelos MLP sobre el dataset Titanic, permitiendo predecir si un pasajero habría sobrevivido o no en base a sus características.

El sistema entrena los modelos automáticamente al iniciarse, evalúa su desempeño sobre datos reales y expone una interfaz de predicción en tiempo real.

---

## 🧠 Tipo de aprendizaje

**Aprendizaje supervisado** — clasificación binaria.

- Variable objetivo: `Survived` (0 = No sobrevivió, 1 = Sobrevivió)
- El modelo aprende a partir de ejemplos etiquetados del dataset histórico

---

## 📂 Dataset

- **Fuente:** [Titanic Dataset — datasciencedojo/datasets (GitHub)](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Registros:** 891 pasajeros
- **Desbalance original:** ~62% fallecidos / ~38% sobrevivientes

---

## ⚙️ Pipeline

### 1. Preprocesamiento

| Técnica | Descripción |
|---|---|
| Selección de variables | Se conservan solo las features con poder predictivo real |
| Imputación | `Age` nulos rellenados con la **mediana** (robusta ante valores extremos) |
| Encoding binario | `Sex`: male→0, female→1 |
| Feature Engineering | `Title` extraído del nombre del pasajero mediante expresión regular |
| Feature Engineering | `FamilySize` = SibSp + Parch (tamaño familiar combinado) |

**Variables utilizadas:** `Pclass`, `Sex`, `Age`, `Title`, `FamilySize`

**Variables descartadas:**
- `Fare` — redundante con Pclass
- `Embarked` — correlación espuria con la supervivencia
- `Name`, `Ticket`, `Cabin`, `PassengerId` — sin valor predictivo directo

---

### 2. Feature Engineering — Título

Se extrae el título social del nombre de cada pasajero (`Mr`, `Mrs`, `Miss`, `Master`, títulos especiales) mediante expresión regular. Los títulos equivalentes se unifican y los poco frecuentes se agrupan en `Rare`. Luego se aplica encoding numérico:

| Título | Valor | Perfil |
|---|---|---|
| Mr | 0 | Hombre adulto — baja supervivencia |
| Miss | 1 | Mujer soltera/joven — alta supervivencia |
| Mrs | 2 | Mujer casada — alta supervivencia |
| Master | 3 | Niño varón < 13 años — alta supervivencia |
| Rare | 4 | Títulos especiales (Dr, Rev, Col, etc.) |

---

### 3. División del dataset

| Partición | Porcentaje | Uso |
|---|---|---|
| Train | 70% | Entrenamiento del modelo |
| Validación | 15% | Monitoreo con early stopping |
| Test | 15% | Evaluación final del modelo |

Se utiliza `stratify=y` para mantener la proporción original de clases en cada partición.

---

### 4. Balanceo — SMOTE

**Técnica:** SMOTE *(Synthetic Minority Over-sampling Technique)*

Genera instancias sintéticas de la clase minoritaria (Survived=1) interpolando entre ejemplos reales vecinos en el espacio de features. Se aplica **exclusivamente sobre el conjunto de train** para no contaminar la evaluación.

---

### 5. Escalado — StandardScaler

Transforma cada feature para que tenga **media = 0 y desvío estándar = 1**, poniendo todas las variables en la misma escala. El scaler se entrena solo con los datos de train y luego se aplica a validación y test.

---

## 🧠 Modelos MLP

Se construyeron y compararon 3 arquitecturas distintas de Multilayer Perceptron:

### Modelo 1 — Sigmoid
| Hiperparámetro | Valor |
|---|---|
| Capas ocultas | 1 → (64 neuronas) |
| Función de activación | Sigmoid (logistic) |
| Learning rate | 0.001 |
| Épocas máximas | 50 |
| Regularización L2 | — |
| Early stopping | Sí |

Sigmoid es válida en redes poco profundas donde el vanishing gradient no representa un problema crítico.

---

### Modelo 2 — ReLU
| Hiperparámetro | Valor |
|---|---|
| Capas ocultas | 3 → (128 → 64 → 32) |
| Función de activación | ReLU |
| Learning rate | 0.001 |
| Épocas máximas | 100 |
| Regularización L2 | alpha = 0.01 |
| Early stopping | Sí |

Arquitectura en patrón embudo. ReLU evita el vanishing gradient en redes profundas. Regularización L2 para reducir overfitting.

---

### Modelo 3 — Tanh
| Hiperparámetro | Valor |
|---|---|
| Capas ocultas | 2 → (128 → 64) |
| Función de activación | Tanh |
| Learning rate | 0.0005 |
| Épocas máximas | 150 |
| Regularización L2 | alpha = 0.005 |
| Early stopping | Sí |

Tanh es ideal con datos centrados en 0 (resultado del StandardScaler). Learning rate reducido para convergencia más estable, compensado con más épocas.

---

## 📊 Métricas de evaluación

| Métrica | Descripción |
|---|---|
| Accuracy | % de predicciones correctas sobre el total |
| Precision | De los predichos como "sobrevivió", ¿cuántos realmente sobrevivieron? |
| Recall | De los que realmente sobrevivieron, ¿cuántos detectó el modelo? |
| F1-Score | Promedio armónico entre Precision y Recall — **criterio principal** |
| AUC-ROC | Capacidad discriminativa entre clases (0.5=aleatorio, 1.0=perfecto) |
| Matriz de confusión | Detalle de VP, VN, FP y FN por modelo |

**¿Por qué F1-Score como criterio principal?**
El set de test conserva la distribución real desbalanceada del dataset (~62/38). En ese contexto, la Accuracy puede ser engañosa — un modelo que siempre prediga "no sobrevivió" tendría 62% de Accuracy sin haber aprendido nada. El F1-Score penaliza tanto los falsos positivos como los falsos negativos, siendo más representativo del desempeño real.

---

## 📈 Visualizaciones

- Gráfico de barras comparativo (Accuracy, Precision, Recall, F1)
- Radar multimétrica
- Curvas de pérdida (Loss) por época
- Curvas ROC con AUC
- Matrices de confusión

---

## 🔮 Interfaz de predicción

Permite ingresar los datos de un pasajero y obtener la predicción de los 3 modelos en tiempo real. El título se infiere automáticamente a partir del sexo y la edad para garantizar coherencia entre variables:

```
Masculino + edad < 13  →  Master
Masculino + edad ≥ 13  →  Mr
Femenino  + familia > 0 →  Mrs
Femenino  + familia = 0 →  Miss
```

---

## 🛠️ Tecnologías utilizadas

| Librería | Uso |
|---|---|
| `streamlit` | Interfaz web interactiva |
| `pandas` | Carga y manipulación del dataset |
| `numpy` | Operaciones numéricas y vectores |
| `scikit-learn` | Modelos MLP, StandardScaler, métricas, split |
| `imbalanced-learn` | SMOTE para balanceo de clases |
| `plotly` | Gráficos interactivos |
| `matplotlib` | Matrices de confusión |
| `seaborn` | Heatmap de matrices de confusión |

---

## 🚀 Instalación y ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run rn_titanic.py
```

### requirements.txt
```
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
plotly
```

---

## 📁 Estructura del proyecto

```
├── rn_titanic.py       # Aplicación principal
├── requirements.txt    # Dependencias
└── README.md           # Este archivo
```

---

## 👥 Equipo

TP5 — Inteligencia Artificial 2026
