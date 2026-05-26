import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Titanic · MLP Classifier",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #e8edf4; }
    .stMetric { background-color: #0f1923; border-radius: 10px; padding: 8px;
                border: 1px solid #1e3050; }
    div[data-testid="stExpander"] { border: 1px solid #1e3050 !important;
                                    border-radius: 10px !important; }
    .stTabs [data-baseweb="tab"] { color: #8a9bb0; }
    .stTabs [aria-selected="true"] { color: #4F8EF7; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PIPELINE — cacheado para no re-entrenar con cada interacción del usuario
# =============================================================================

@st.cache_data
def cargar_y_procesar():
    """Carga el dataset y aplica todo el preprocesamiento."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df  = pd.read_csv(url)

    df_model = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Name", "Survived"]].copy()

    # ── Feature engineering: Title extraído del nombre ───────────────────────
    # Extraemos el título (palabra entre coma y punto)
    df_model["Title"] = df_model["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Unificamos títulos equivalentes
    df_model["Title"] = df_model["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Títulos poco frecuentes → agrupados en "Rare"
    titulos_rare = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
    df_model["Title"] = df_model["Title"].replace(titulos_rare, "Rare")

    # Encoding numérico:
    df_model["Title"] = df_model["Title"].map(
        {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    )
    df_model["Title"] = df_model["Title"].fillna(0)

    # Eliminamos Name — ya extrajimos lo que necesitábamos
    df_model.drop(columns=["Name"], inplace=True)
    # ─────────────────────────────────────────────────────────────────────────

    # FamilySize = SibSp + Parch combinados
    df_model["FamilySize"] = df_model["SibSp"] + df_model["Parch"]
    df_model.drop(columns=["SibSp", "Parch"], inplace=True)

    # Imputación de Age con mediana (asignación directa — compatible con pandas moderno)
    df_model["Age"] = df_model["Age"].fillna(df_model["Age"].median())

    # Encoding: male→0, female→1
    df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})

    X = df_model.drop(columns=["Survived"])
    y = df_model["Survived"]
    return X, y


@st.cache_data
def preparar_datos(_X, _y):
    """
    Divide en 70% train / 15% validación / 15% test,
    aplica SMOTE solo sobre train y escala los datos.
    """
    # Garantizamos que no haya NaN antes de SMOTE
    _X = _X.copy()
    for col in _X.columns:
        if _X[col].isnull().any():
            _X[col] = _X[col].fillna(_X[col].median())

    # Split 70% / 15% / 15%  — stratify mantiene proporción de clases en cada partición
    X_temp, X_test, y_temp, y_test = train_test_split(
        _X, _y, test_size=0.15, random_state=42, stratify=_y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    # SMOTE solo sobre train — nunca sobre val ni test
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # StandardScaler — fit solo sobre train, transform en val y test
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_bal)
    X_val_sc    = scaler.transform(X_val)
    X_test_sc   = scaler.transform(X_test)

    sizes = {
        "train":      len(X_train),
        "train_smote": len(X_train_bal),
        "val":        len(X_val),
        "test":       len(X_test),
        "total":      len(_X),
    }
    return X_train_sc, X_val_sc, X_test_sc, y_train_bal, y_val, y_test, scaler, sizes


@st.cache_data
def entrenar_y_evaluar(_X_train_sc, _y_train_bal, _X_test_sc, _y_test):
    """Entrena los 3 modelos MLP y devuelve métricas completas."""

    modelos_def = {
        "Modelo 1 — Sigmoid": MLPClassifier(
            hidden_layer_sizes=(64,),       # 1 capa oculta, 64 neuronas
            activation="logistic",           # Sigmoid: válida en redes poco profundas
            learning_rate_init=0.001,
            max_iter=50,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
        "Modelo 2 — ReLU": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), # 3 capas ocultas: patrón embudo
            activation="relu",                # ReLU: evita vanishing gradient en redes profundas
            learning_rate_init=0.001,
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.01                        # Regularización L2
        ),
        "Modelo 3 — Tanh": MLPClassifier(
            hidden_layer_sizes=(128, 64),     # 2 capas ocultas: patrón embudo
            activation="tanh",                # Tanh: ideal con datos centrados en 0
            learning_rate_init=0.0005,
            max_iter=150,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.005
        ),
    }

    resultados = {}
    for nombre, modelo in modelos_def.items():
        t0     = time.time()
        modelo.fit(_X_train_sc, _y_train_bal)
        tiempo = round(time.time() - t0, 2)

        y_pred      = modelo.predict(_X_test_sc)
        y_pred_prob = modelo.predict_proba(_X_test_sc)[:, 1]

        resultados[nombre] = {
            "modelo":    modelo,
            "accuracy":  round(accuracy_score(_y_test, y_pred) * 100, 2),
            "precision": round(precision_score(_y_test, y_pred) * 100, 2),
            "recall":    round(recall_score(_y_test, y_pred) * 100, 2),
            "f1":        round(f1_score(_y_test, y_pred) * 100, 2),
            "auc":       round(roc_auc_score(_y_test, y_pred_prob), 3),
            "tiempo":    tiempo,
            "epocas":    modelo.n_iter_,
            "cm":        confusion_matrix(_y_test, y_pred),
            "loss":      modelo.loss_curve_,
            "y_prob":    y_pred_prob,
        }

    return resultados


# =============================================================================
# ENTRENAMIENTO AUTOMÁTICO AL INICIAR
# =============================================================================
with st.spinner("⏳ Cargando dataset y entrenando modelos..."):
    X, y = cargar_y_procesar()
    X_train_sc, X_val_sc, X_test_sc, y_train_bal, y_val, y_test, scaler, sizes = preparar_datos(X, y)
    resultados = entrenar_y_evaluar(X_train_sc, y_train_bal, X_test_sc, y_test)

COLORES = {
    "Modelo 1 — Sigmoid": "#00C9A7",
    "Modelo 2 — ReLU":    "#4F8EF7",
    "Modelo 3 — Tanh":    "#F7B74F",
}
NOMBRES_CORTOS = {
    "Modelo 1 — Sigmoid": "M1 · Sigmoid",
    "Modelo 2 — ReLU":    "M2 · ReLU",
    "Modelo 3 — Tanh":    "M3 · Tanh",
}
mejor_nombre = max(resultados, key=lambda k: resultados[k]["f1"])


# =============================================================================
# SIDEBAR — configuración del pipeline y modelos
# =============================================================================
with st.sidebar:
    st.markdown("## 📋 Configuración del pipeline")

    st.markdown("### 📂 Dataset")
    st.markdown(f"""
    - **Total de registros:** {sizes['total']}
    - **Variables usadas:** Pclass, Sex, Age, Title, FamilySize
    - **Creadas:** Title *(extraído del nombre)*, FamilySize *(SibSp + Parch)*
    """)

    st.markdown("---")
    st.markdown("### ✂️ División del dataset")
    st.markdown(f"""
    | Partición | % | Registros |
    |---|---|---|
    | Train | 70% | {sizes['train']} |
    | Validación | 15% | {sizes['val']} |
    | Test | 15% | {sizes['test']} |

    *Tras aplicar SMOTE el train pasa a {sizes['train_smote']} registros (balanceado 50/50)*
    """)

    st.markdown("---")
    st.markdown("### ⚖️ Balanceo")
    st.markdown("""
    **Técnica:** SMOTE *(Synthetic Minority Over-sampling)*

    Genera instancias sintéticas de la clase minoritaria
    *(Survived=1)* interpolando entre ejemplos reales vecinos.
    """)

    st.markdown("---")
    st.markdown("### 🧠 Arquitectura de los modelos")

    for nombre, color in COLORES.items():
        modelo = resultados[nombre]["modelo"]
        st.markdown(f"<span style='color:{color}; font-weight:700'>**{nombre}**</span>",
                    unsafe_allow_html=True)

        if nombre == "Modelo 1 — Sigmoid":
            st.markdown("""
            - Capas ocultas: **1** → (64 neuronas)
            - Activación: **Sigmoid** *(logistic)*
            - Learning rate: **0.001**
            - Épocas máx: **50**
            - Regularización: **—**
            """)
        elif nombre == "Modelo 2 — ReLU":
            st.markdown("""
            - Capas ocultas: **3** → (128 → 64 → 32)
            - Activación: **ReLU**
            - Learning rate: **0.001**
            - Épocas máx: **100**
            - Regularización L2: **alpha=0.01**
            """)
        else:
            st.markdown("""
            - Capas ocultas: **2** → (128 → 64)
            - Activación: **Tanh**
            - Learning rate: **0.0005**
            - Épocas máx: **150**
            - Regularización L2: **alpha=0.005**
            """)
        st.markdown("---")



# =============================================================================
# ENCABEZADO
# =============================================================================
st.title("🚢 Titanic · MLP Classifier")
st.markdown(
    "**TP5 · RNA No Simbólica · Multilayer Perceptron** "
    "— Completá los datos del pasajero y los 3 modelos predicen si sobrevivió."
)
st.markdown("---")


# =============================================================================
# SECCIÓN PRINCIPAL — PREDICCIÓN
# =============================================================================
st.subheader("🔮 Predecir supervivencia")

col_form, col_sep, col_resultado = st.columns([1.1, 0.05, 1.8])

with col_form:
    st.markdown("##### Datos del pasajero")

    pclass = st.selectbox(
        "Clase",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1ra clase", 2: "2da clase", 3: "3ra clase"}[x],
        help="Determina la cubierta y el acceso a los botes"
    )

    sex = st.radio(
        "Sexo",
        options=["female", "male"],
        format_func=lambda x: "Femenino" if x == "female" else "Masculino",
        horizontal=True,
        help="Variable de mayor impacto en la supervivencia"
    )

    age = st.slider(
        "Edad", min_value=0, max_value=80, value=28,
        help="Los niños tenían prioridad en la evacuación"
    )

    family_size = st.slider(
        "Tamaño familiar",
        min_value=0, max_value=10, value=0,
        help="Total de familiares a bordo (SibSp + Parch)"
    )

    # Título inferido automáticamente según sexo y edad
    # (no se muestra como campo en el formulario)
    if sex == "male":
        title = 3 if age < 13 else 0   # Master si es niño, Mr si es adulto
    else:
        # Femenino: Mrs si tiene familia, Miss si está sola
        title = 2 if family_size > 0 else 1

    # Mostrar el título inferido como campo informativo (solo lectura)
    TITULO_LABELS = {0: "Mr", 1: "Miss", 2: "Mrs", 3: "Master", 4: "Rare"}
    titulo_label  = TITULO_LABELS[title]

    st.markdown("**Título inferido**")
    st.markdown(f"""
    <div style='background:#0b1520; border:1px solid #1e3050; border-radius:8px;
                padding:9px 12px; font-size:14px; color:#e8edf4; margin-bottom:16px'>
        🏷️ <strong style='color:#4F8EF7'>{titulo_label}</strong>
        <span style='color:#5a6a80; font-size:12px; margin-left:8px'>
        — inferido a partir de sexo y edad
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_resultado:
    st.markdown("##### Resultado por modelo")

    # Vector de features — mismo orden que el entrenamiento:
    # [Pclass, Sex, Age, Title, FamilySize]
    sex_num  = 1 if sex == "female" else 0
    datos_sc = scaler.transform(np.array([[pclass, sex_num, age, title, family_size]]))

    # Los 3 modelos se presentan de forma equitativa — sin indicar cuál es mejor
    for nombre, vals in resultados.items():
        prob       = vals["modelo"].predict_proba(datos_sc)[0][1]
        sobrevive  = prob >= 0.5
        diferencia = abs(prob - 0.5)
        confianza  = "Alta" if diferencia > 0.3 else ("Media" if diferencia > 0.15 else "Baja")
        color      = COLORES[nombre]
        icono_conf = "🟢" if diferencia > 0.3 else ("🟡" if diferencia > 0.15 else "🔴")

        st.markdown(f"""
        <div style='background:#0f1923;
                    border:1px solid {"#00C9A744" if sobrevive else "#ff4d4d44"};
                    border-radius:12px; padding:14px 18px; margin-bottom:10px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px'>
                <span style='color:{color}; font-weight:700; font-size:15px'>{nombre}</span>
                <span style='color:{"#00C9A7" if sobrevive else "#ff6b6b"};
                             font-weight:700; font-size:16px'>
                    {"✅ Sobrevive" if sobrevive else "❌ No sobrevive"}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:5px'>
                <span style='color:#8a9bb0; font-size:12px'>Probabilidad de sobrevivir</span>
                <span style='color:{color}; font-weight:700; font-size:13px'>{round(prob*100,1)}%</span>
            </div>
            <div style='height:9px; background:#1a2a3a; border-radius:5px; overflow:hidden; margin-bottom:6px'>
                <div style='height:100%; width:{round(prob*100,1)}%;
                            background:{color}; border-radius:5px;'></div>
            </div>
            <span style='color:#8a9bb0; font-size:12px'>Confianza: {icono_conf} {confianza}</span>
        </div>
        """, unsafe_allow_html=True)

    titulo_label = {0: "Mr", 1: "Miss", 2: "Mrs", 3: "Master", 4: "Rare"}[title]
    st.markdown(f"""
    <div style='background:#0b1520; border:1px solid #1e3050; border-radius:8px;
                padding:8px 14px; margin-top:4px; font-size:12px; color:#8a9bb0'>
        Pasajero evaluado: <strong style='color:#cdd8e3'>
        {'Femenino' if sex == 'female' else 'Masculino'} ·
        {titulo_label} · Clase {pclass} · {age} años · Familia: {family_size}
        </strong>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SECCIÓN 2 — MÉTRICAS COMPARATIVAS
# =============================================================================
st.markdown("---")
with st.expander("📊  Ver métricas comparativas de los 3 modelos", expanded=False):

    st.markdown(
        "Resultados reales del entrenamiento evaluados sobre el **set de test** (15% del dataset)."
    )

    tabla = pd.DataFrame({
        NOMBRES_CORTOS[n]: {
            "Accuracy (%)":  v["accuracy"],
            "Precision (%)": v["precision"],
            "Recall (%)":    v["recall"],
            "F1-Score (%)":  v["f1"],
            "AUC-ROC":       v["auc"],
            "Épocas reales": v["epocas"],
            "Tiempo (s)":    v["tiempo"],
        }
        for n, v in resultados.items()
    }).T

    st.dataframe(tabla.style.format("{:.2f}"), use_container_width=True)

    csv = tabla.to_csv().encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, "resultados_modelos.csv", "text/csv")


# =============================================================================
# SECCIÓN 3 — GRÁFICOS
# =============================================================================
with st.expander("📈  Ver gráficos comparativos", expanded=False):

    tab_barras, tab_radar, tab_loss, tab_roc, tab_cm = st.tabs([
        "Barras", "Radar", "Curva Loss", "Curva ROC", "Matrices de confusión"
    ])

    with tab_barras:
        mkeys   = ["accuracy", "precision", "recall", "f1"]
        mlabels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        fig = go.Figure()
        for nombre, vals in resultados.items():
            fig.add_trace(go.Bar(
                name=NOMBRES_CORTOS[nombre],
                x=mlabels,
                y=[vals[m] for m in mkeys],
                marker_color=COLORES[nombre],
                opacity=0.85,
            ))
        fig.update_layout(
            barmode="group", height=350,
            plot_bgcolor="#0f1923", paper_bgcolor="#0f1923", font_color="#e8edf4",
            yaxis=dict(range=[60, 100], gridcolor="#1e3050"),
            xaxis=dict(gridcolor="#1e3050"),
            legend=dict(bgcolor="#0b1520", bordercolor="#1e3050"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_radar:
        cats = ["Accuracy", "Precision", "Recall", "F1", "AUC×100"]
        fig  = go.Figure()
        for nombre, vals in resultados.items():
            vs = [vals["accuracy"], vals["precision"],
                  vals["recall"], vals["f1"], vals["auc"] * 100]
            fig.add_trace(go.Scatterpolar(
                r=vs + [vs[0]], theta=cats + [cats[0]],
                fill="toself", name=NOMBRES_CORTOS[nombre],
                line_color=COLORES[nombre], fillcolor=COLORES[nombre], opacity=0.2,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="#0f1923",
                radialaxis=dict(visible=True, range=[60, 100], gridcolor="#1e3050", color="#8a9bb0"),
                angularaxis=dict(gridcolor="#1e3050", color="#8a9bb0"),
            ),
            paper_bgcolor="#0f1923", font_color="#e8edf4", height=380,
            legend=dict(bgcolor="#0b1520", bordercolor="#1e3050"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_loss:
        fig = go.Figure()
        for nombre, vals in resultados.items():
            fig.add_trace(go.Scatter(
                y=vals["loss"], x=list(range(1, len(vals["loss"]) + 1)),
                mode="lines", name=NOMBRES_CORTOS[nombre],
                line=dict(color=COLORES[nombre], width=2),
            ))
        fig.update_layout(
            height=340, plot_bgcolor="#0f1923", paper_bgcolor="#0f1923", font_color="#e8edf4",
            xaxis=dict(title="Época", gridcolor="#1e3050"),
            yaxis=dict(title="Loss",  gridcolor="#1e3050"),
            legend=dict(bgcolor="#0b1520", bordercolor="#1e3050"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_roc:
        fig = go.Figure()
        for nombre, vals in resultados.items():
            fpr, tpr, _ = roc_curve(y_test, vals["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{NOMBRES_CORTOS[nombre]} (AUC={vals['auc']})",
                line=dict(color=COLORES[nombre], width=2),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio",
            line=dict(color="#4a5568", dash="dash"),
        ))
        fig.update_layout(
            height=380, plot_bgcolor="#0f1923", paper_bgcolor="#0f1923", font_color="#e8edf4",
            xaxis=dict(title="Falsos Positivos", gridcolor="#1e3050"),
            yaxis=dict(title="Verdaderos Positivos", gridcolor="#1e3050"),
            legend=dict(bgcolor="#0b1520", bordercolor="#1e3050"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_cm:
        cols = st.columns(3)
        for i, (nombre, vals) in enumerate(resultados.items()):
            with cols[i]:
                st.markdown(
                    f"<span style='color:{COLORES[nombre]}; font-weight:700'>"
                    f"{NOMBRES_CORTOS[nombre]}</span>", unsafe_allow_html=True
                )
                cm     = vals["cm"]
                fig_cm, ax = plt.subplots(figsize=(4, 3))

                # Fondo blanco para que el colormap tenga buen contraste
                fig_cm.patch.set_facecolor("white")
                ax.set_facecolor("white")

                sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False,
                            xticklabels=["Pred. No", "Pred. Sí"],
                            yticklabels=["Real No",  "Real Sí"],
                            annot_kws={"size": 14, "color": "black", "weight": "bold"},
                            linewidths=1, linecolor="#cccccc")

                ax.tick_params(colors="black")
                ax.set_xlabel("Predicho", color="black", fontsize=11)
                ax.set_ylabel("Real",     color="black", fontsize=11)
                ax.set_title(NOMBRES_CORTOS[nombre], color="black", fontsize=11, pad=8)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()
                vn, fp, fn, vp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                st.caption(f"VP={vp} · VN={vn} · FP={fp} · FN={fn}")


# =============================================================================
# SECCIÓN 4 — MEJOR MODELO (automático)
# =============================================================================
with st.expander("🏆  Mejor modelo", expanded=False):

    mejor_vals  = resultados[mejor_nombre]
    mejor_color = COLORES[mejor_nombre]

    st.success(f"**{mejor_nombre}** es el modelo con mejor desempeño según F1-Score.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{mejor_vals['accuracy']}%")
    c2.metric("F1-Score",  f"{mejor_vals['f1']}%")
    c3.metric("AUC-ROC",   f"{mejor_vals['auc']}")
    c4.metric("Tiempo",    f"{mejor_vals['tiempo']}s")

    st.markdown("---")
    st.markdown("#### ¿Por qué F1-Score como criterio principal?")
    st.info("""
    El dataset Titanic original tiene ~62% de fallecidos y ~38% de sobrevivientes.
    Aunque aplicamos **SMOTE para balancear el entrenamiento**, el set de test conserva
    la distribución real — porque es lo que el modelo va a encontrar en el mundo real.

    Evaluar sobre datos reales desbalanceados significa que la **Accuracy puede ser engañosa**:
    un modelo que siempre prediga "no sobrevivió" tendría 62% de Accuracy sin haber aprendido nada.

    El **F1-Score** combina Precision y Recall, penalizando tanto los falsos positivos
    como los falsos negativos. Es mucho más representativo del desempeño real en este caso.
    """)
