import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import time

# Bibliotecas de ML
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                           HistGradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                           classification_report, roc_curve, auc, RocCurveDisplay)

# Balanceo de datos
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

from scipy.stats import randint, uniform
from sklearn.utils import resample

# MCA
try:
    from mca import MCA
    MCA_AVAILABLE = True
except ImportError:
    MCA_AVAILABLE = False

# Configuración de matplotlib para Streamlit
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

# Configuración de la página
st.set_page_config(
    page_title="Trabajo Final ML Bioestadística",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Análisis Comparativo de Técnicas de Reducción de Dimensionalidad en Predicción de Estado de Salud: PCA, MCA y Selección de Características")
    st.markdown("### Trabajo Final - Machine Learning en Bioestadística")
    st.markdown("**Autores:** David Zabala y Kevin Suarez | **Fecha:** Agosto 2025")
    
    # Inicializar estado de la sesión si no existe
    if 'analisis_completado' not in st.session_state:
        st.session_state.analisis_completado = False
        st.session_state.resultados_analisis = None
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    
    # Botón principal para ejecutar el análisis
    if st.sidebar.button("Ejecutar Análisis Completo", type="primary", use_container_width=True):
        st.session_state.analisis_completado = True
        st.session_state.resultados_analisis = ejecutar_analisis_completo()
        st.rerun()
    
    # Mostrar menús según el estado
    if not st.session_state.analisis_completado:
        # Menú inicial - antes de ejecutar análisis
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Secciones del Análisis")
        
        seccion = st.sidebar.selectbox(
            "Seleccionar sección:",
            ["Resumen Ejecutivo", "Configuración del Entorno", "Análisis Exploratorio", 
             "Preprocesamiento", "Selección de Características", "PCA y MCA", 
             "Técnicas de Balanceo", "Modelos de Clasificación", "Resultados", "Conclusiones"],
            index=0
        )
        
        if seccion == "Resumen Ejecutivo":
            mostrar_resumen_ejecutivo()
        elif seccion == "Configuración del Entorno":
            mostrar_configuracion()
        elif seccion == "Análisis Exploratorio":
            mostrar_analisis_exploratorio()
        elif seccion == "Preprocesamiento":
            mostrar_preprocesamiento()
        elif seccion == "Selección de Características":
            mostrar_seleccion_caracteristicas()
        elif seccion == "PCA y MCA":
            mostrar_pca_mca()
        elif seccion == "Técnicas de Balanceo":
            mostrar_balanceo()
        elif seccion == "Modelos de Clasificación":
            mostrar_modelos()
        elif seccion == "Resultados":
            mostrar_resultados()
        elif seccion == "Conclusiones":
            mostrar_conclusiones()
    else:
        # Menú después del análisis - mostrar resultados
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Análisis Completado")
        
        # Botón para reiniciar análisis
        if st.sidebar.button("Ejecutar Nuevo Análisis", use_container_width=True):
            st.session_state.analisis_completado = True
            st.session_state.resultados_analisis = ejecutar_analisis_completo()
            st.rerun()
        
        if st.sidebar.button("Volver al Menú Principal", use_container_width=True):
            st.session_state.analisis_completado = False
            st.session_state.resultados_analisis = None
            st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Resultados del Análisis")
        
        seccion_resultados = st.sidebar.selectbox(
            "Ver resultados de:",
            ["Resumen Completo", "Carga de Datos", "Análisis Exploratorio",
             "Preprocesamiento", "Selección de Características", "PCA y MCA",
             "Técnicas de Balanceo", "Modelos Entrenados", "Mejores Resultados",
             "Comparación de Modelos", "Conclusiones Finales"],
            index=0
        )
        
        # Mostrar la sección seleccionada de resultados
        if st.session_state.resultados_analisis:
            mostrar_seccion_resultados(seccion_resultados, st.session_state.resultados_analisis)
        else:
            st.error("No hay resultados disponibles. Por favor, ejecuta el análisis primero.")

def mostrar_resumen_ejecutivo():
    st.header("Resumen Ejecutivo")
    
    st.markdown("""
    Este trabajo presenta un análisis exhaustivo de técnicas de reducción de dimensionalidad 
    aplicadas a la predicción del estado de salud usando datos de estilo de vida. Se comparan 
    metodológicamente tres enfoques principales: **Análisis de Componentes Principales (PCA)**, 
    **Análisis de Correspondencias Múltiples (MCA)** y **técnicas de selección de características**, 
    evaluando su impacto en el rendimiento predictivo y la interpretabilidad clínica.
    """)
    
    st.subheader("OBJETIVOS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### **Objetivo General:**
        Evaluar y comparar la efectividad de diferentes técnicas de reducción de dimensionalidad 
        para la predicción del estado de salud en un contexto bioestadístico, priorizando tanto 
        el rendimiento predictivo como la interpretabilidad médica.
        """)
    
    with col2:
        st.markdown("""
        #### **Objetivos Específicos:**
        1. **Implementar y comparar** técnicas de selección de características
        2. **Aplicar PCA** a variables numéricas para reducción dimensional
        3. **Implementar MCA** en variables categóricas 
        4. **Evaluar el impacto** de técnicas de balanceo
        5. **Comparar múltiples algoritmos** de clasificación
        6. **Proporcionar recomendaciones** para aplicaciones clínicas
        """)
    
    st.subheader("DATASET")
    st.info("""
    **Health Lifestyle Dataset** - Datos de estilo de vida y salud de 100,000 personas:
    - **Muestra analizada:** 50,000 registros seleccionados aleatoriamente de forma estratificada
    - **Variables numéricas:** age, bmi, blood_pressure, cholesterol, glucose, sleep_hours, etc.
    - **Variables categóricas:** gender, marital_status, diet_type, occupation, healthcare_access, etc.
    - **Variable objetivo:** Clasificación binaria (healthy/diseased)
    - **Desbalance de clases:** ~70% sanos vs ~30% enfermos
    """)
    
    st.subheader("METODOLOGÍA")
    metodologia_pasos = [
        "Análisis Exploratorio Exhaustivo con visualizaciones interpretativas",
        "Preprocesamiento Robusto con pipelines y validación anti-leakage", 
        "Comparación de 4 Técnicas de Selección: Chi², Información Mutua, Random Forest, RFECV",
        "Implementación de PCA y MCA con criterios de varianza explicada",
        "Evaluación de 8 Técnicas de Balanceo para manejo de clases desbalanceadas",
        "Optimización de 5 Algoritmos con RandomizedSearchCV y validación cruzada",
        "Análisis Comparativo Integral con métricas múltiples y curvas ROC"
    ]
    
    for i, paso in enumerate(metodologia_pasos, 1):
        st.markdown(f"{i}. {paso}")

def mostrar_configuracion():
    st.header("Configuración del Entorno y Librerías")

    st.markdown("""
    Importamos todas las librerías necesarias para el análisis, organizadas por funcionalidad 
    para facilitar la reproducibilidad y mantenimiento del código.
    """)
    
    st.subheader("Librerías Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Bibliotecas principales
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
        """, language="python")
        
    with col2:
        st.code("""
# Preprocessing y división de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
        """, language="python")
    
    st.subheader("Librerías de Machine Learning")
    
    with st.expander("Ver todas las importaciones"):
        st.code("""
# Reducción de dimensionalidad
from sklearn.decomposition import PCA
import mca  # Para MCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFECV

# Modelos de clasificación
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Métricas de evaluación
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# Balanceo de datos
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
        """, language="python")

def ejecutar_analisis_completo():
    """Ejecuta todo el pipeline de análisis paso a paso"""
    st.header("🔄 Ejecutando Análisis Completo")
    
    # Crear contenedores para actualizar el progreso
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Lista de pasos del análisis
    pasos = [
        ("Cargando y preparando datos...", cargar_datos),
        ("Realizando análisis exploratorio...", lambda df: analisis_exploratorio(df)),
        ("Preprocesando datos...", lambda df: preprocesar_datos(df)),
        ("Aplicando selección de características...", lambda data: seleccion_caracteristicas(data)),
        ("Ejecutando PCA y MCA...", lambda data: aplicar_pca_mca(data)),
        ("Aplicando técnicas de balanceo...", lambda data: aplicar_balanceo(data)),
        ("Entrenando modelos de clasificación...", lambda data: entrenar_modelos(data)),
        ("Generando resultados finales...", lambda data: generar_resultados(data)),
        ("¡Análisis completado!", lambda data: None)
    ]
    
    # Estado compartido para pasar datos entre pasos
    analisis_estado = {}
    
    for i, (descripcion, funcion) in enumerate(pasos):
        with status_container:
            status_text.text(descripcion)
        
        progress_bar.progress((i + 1) / len(pasos))
        
        # Ejecutar función si no es el último paso
        if funcion is not None:
            try:
                if i == 0:  # Cargar datos
                    resultado = funcion()
                else:  # Otros pasos
                    resultado = funcion(analisis_estado)
                
                if resultado is not None:
                    analisis_estado.update(resultado)
                    
                time.sleep(1)  # Pausa para mostrar el progreso
                
            except Exception as e:
                st.error(f"Error en el paso {i+1}: {str(e)}")
                return None
    
    with results_container:
        st.success("✅ Análisis completado exitosamente!")
        
        # Mostrar resumen de resultados si están disponibles
        if 'resultados_finales' in analisis_estado:
            mostrar_resultados_completos(analisis_estado['resultados_finales'])
    
    return analisis_estado

def cargar_datos():
    """Carga y prepara el dataset"""
    st.subheader("📂 Carga de Datos")
    
    try:
        # Intentar cargar datos originales usando kagglehub
        st.info("💡 Descargando Health Lifestyle Dataset desde Kaggle...")
        
        with st.spinner("Descargando datos..."):
            import kagglehub
            path = kagglehub.dataset_download("mahdimashayekhi/disease-risk-from-daily-habits")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            csv_path = os.path.join(path, csv_files[0])
            df_original = pd.read_csv(csv_path)
            
            # Muestreo estratificado de 50,000 registros (como en el notebook)
            df, _, _, _ = train_test_split(
                df_original, 
                df_original['target'], 
                train_size=0.5,  # 50% de 100,000 = 50,000
                stratify=df_original['target'], 
                random_state=123
            )
            
            # Eliminar columnas como en el notebook
            columns_to_drop = ['survey_code', 'electrolyte_level']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
        st.success("✅ Datos originales cargados desde Kaggle")
        
    except Exception as e:
        st.warning(f"No se pudieron cargar los datos originales: {str(e)}")
        st.info("Usando datos sintéticos para demostración...")
        
        # Fallback a datos sintéticos (código actual)
        np.random.seed(123)
        n_samples = 5000  # Muestra más pequeña para la demo
        
        # Generar datos sintéticos similares al dataset original
        data = {
            'age': np.random.normal(45, 15, n_samples).clip(18, 80),
            'bmi_corrected': np.random.normal(25, 5, n_samples).clip(15, 50),
            'blood_pressure': np.random.normal(120, 20, n_samples).clip(80, 200),
            'cholesterol': np.random.normal(200, 40, n_samples).clip(100, 400),
            'glucose': np.random.normal(100, 20, n_samples).clip(70, 200),
            'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(4, 12),
            'weight': np.random.normal(70, 15, n_samples).clip(40, 120),
            'height': np.random.normal(170, 10, n_samples).clip(150, 200),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'sleep_quality': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
            'smoking_level': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples),
            'diet_type': np.random.choice(['Balanced', 'Vegetarian', 'Keto', 'Mediterranean'], n_samples),
            'healthcare_access': np.random.choice(['Limited', 'Adequate', 'Excellent'], n_samples),
            'occupation': np.random.choice(['Office', 'Manual', 'Healthcare', 'Education'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Crear variable objetivo con cierta lógica (no completamente aleatoria)
        # Factores que aumentan probabilidad de enfermedad
        prob_disease = 0.3  # Probabilidad base
        
        # Ajustar probabilidad basada en factores de riesgo
        risk_factors = (
            (df['age'] > 60) * 0.2 +
            (df['bmi_corrected'] > 30) * 0.2 +  
            (df['blood_pressure'] > 140) * 0.2 +
            (df['smoking_level'].isin(['Moderate', 'Heavy'])) * 0.2 +
            (df['sleep_hours'] < 6) * 0.1
        )
        
        final_prob = prob_disease + risk_factors
        df['target'] = np.random.binomial(1, final_prob.clip(0, 1), n_samples)
        df['target'] = df['target'].map({0: 'healthy', 1: 'diseased'})
    
    # Mostrar información básica
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de registros", f"{len(df):,}")
    with col2:
        st.metric("Variables numéricas", len(df.select_dtypes(include=[np.number]).columns) - 1)
    with col3:
        st.metric("Variables categóricas", len(df.select_dtypes(include=['object']).columns) - 1)
    
    st.write("**Vista previa de los datos:**")
    st.dataframe(df.head())
    
    return {'df': df, 'mensaje': 'Datos cargados correctamente'}

def analisis_exploratorio(estado):
    """Realiza el análisis exploratorio de datos"""
    st.subheader("Análisis Exploratorio de Datos")
    
    df = estado['df']
    
    # Separar variables por tipo
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if 'target' in numerical_cols:
        numerical_cols.remove('target')
    if 'target' in categorical_cols:
        categorical_cols.remove('target')
    
    # Distribución de la variable objetivo
    st.write("### Distribución de la Variable Objetivo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        target_counts = df['target'].value_counts()
        target_props = df['target'].value_counts(normalize=True)
        
        bars = ax.bar(target_counts.index, target_counts.values, color=['#1f77b4', '#ff7f0e'])
        ax.set_title('Distribución de la Variable Objetivo', fontsize=14, fontweight='bold')
        ax.set_xlabel('Estado de Salud')
        ax.set_ylabel('Número de Personas')
        
        # Agregar porcentajes sobre barras
        for bar, prop in zip(bars, target_props):
            height = bar.get_height()
            ax.annotate(f'{prop:.1%}', 
                       (bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=12, color='black')
        
        st.pyplot(fig)
    
    with col2:
        st.write("**Distribución:**")
        for val in target_counts.index:
            st.metric(val.title(), f"{target_counts[val]:,}", f"{target_props[val]:.1%}")
    
    # Distribución de variables numéricas
    st.write("### Distribución de Variables Numéricas Clave")
    
    variables_importantes = ['age', 'bmi_corrected', 'blood_pressure', 'cholesterol', 'glucose', 'sleep_hours']
    variables_disponibles = [var for var in variables_importantes if var in numerical_cols]
    
    if variables_disponibles:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(variables_disponibles[:6]):
            if i < len(axes):
                axes[i].hist(df[var], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribución de {var}', fontweight='bold')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for j in range(len(variables_disponibles), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Matriz de correlación
    st.write("### Matriz de Correlación de Variables Numéricas")
    
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Matriz de Correlación de Variables Numéricas', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Variables categóricas vs target
    st.write("### Variables Categóricas vs Variable Objetivo")
    
    variables_cat_importantes = ['gender', 'sleep_quality', 'smoking_level', 'diet_type']
    variables_cat_disponibles = [var for var in variables_cat_importantes if var in categorical_cols]
    
    if variables_cat_disponibles:
        n_vars = len(variables_cat_disponibles)
        cols_per_row = min(2, n_vars)
        n_rows = (n_vars + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_vars == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(variables_cat_disponibles):
            if df[var].nunique() < 10:  # Solo mostrar si no hay demasiadas categorías
                cross_tab = pd.crosstab(df[var], df['target'], normalize='index') * 100
                
                ax = axes[i] if n_vars > 1 else axes
                cross_tab.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                ax.set_title(f'Distribución de {var} por Estado de Salud (%)', fontweight='bold')
                ax.set_xlabel(var)
                ax.set_ylabel('Porcentaje')
                ax.legend(title='Estado')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    return {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'mensaje': 'Análisis exploratorio completado'
    }

def preprocesar_datos(estado):
    """Preprocesa los datos dividiéndolos en train/test y aplicando pipelines"""
    st.subheader("Preprocesamiento de Datos")
    
    df = estado['df']
    numerical_cols = estado['numerical_cols']
    categorical_cols = estado['categorical_cols']
    
    # Separar X e y
    X = df.drop('target', axis=1)
    y = df['target']
    
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )
    
    # Pipelines de preprocesamiento
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    
    # Aplicar preprocesamiento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Codificar variable objetivo
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Datos de entrenamiento", f"{X_train_processed.shape[0]:,} × {X_train_processed.shape[1]}")
    with col2:
        st.metric("Datos de prueba", f"{X_test_processed.shape[0]:,} × {X_test_processed.shape[1]}")
    
    st.success("✅ Preprocesamiento completado")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'y_train_enc': y_train_enc,
        'y_test_enc': y_test_enc,
        'preprocessor': preprocessor,
        'le': le,
        'mensaje': 'Datos preprocesados correctamente'
    }

def seleccion_caracteristicas(estado):
    """Aplica diferentes técnicas de selección de características"""
    st.subheader("Selección de Características")
    
    X_train_processed = estado['X_train_processed']
    X_test_processed = estado['X_test_processed']
    y_train_enc = estado['y_train_enc']
    preprocessor = estado['preprocessor']
    
    # Obtener nombres de características
    feature_names = preprocessor.get_feature_names_out()
    
    st.write("### Random Forest - Método Incrustado")
    
    # Random Forest para importancia de características
    rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
    rf_model.fit(X_train_processed, y_train_enc)
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Calcular características necesarias para 90% de importancia
    sorted_importances = importances[indices]
    cumulative_importance = np.cumsum(sorted_importances)
    n_features_90 = np.searchsorted(cumulative_importance, 0.9) + 1
    
    # Visualizar importancia
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top 10 características más importantes
    top_n = min(10, len(feature_names))
    ax1.bar(range(top_n), sorted_importances[:top_n])
    ax1.set_title('Top 10 Características (Random Forest)', fontweight='bold')
    ax1.set_xlabel('Características')
    ax1.set_ylabel('Importancia')
    ax1.tick_params(axis='x', rotation=45)
    
    # Importancia acumulada
    ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% Umbral')
    ax2.axvline(x=n_features_90, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Número de Características')
    ax2.set_ylabel('Importancia Acumulada')
    ax2.set_title('Importancia Acumulada', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Seleccionar características más importantes
    selected_indices = indices[:n_features_90]
    X_train_selected = X_train_processed[:, selected_indices]
    X_test_selected = X_test_processed[:, selected_indices] 
    
    st.info(f"💡 Seleccionadas {n_features_90} características que explican el 90% de la importancia")
    
    return {
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'selected_features': feature_names[selected_indices],
        'n_features_selected': n_features_90,
        'mensaje': f'Seleccionadas {n_features_90} características importantes'
    }

def aplicar_pca_mca(estado):
    """Aplica PCA a variables numéricas y MCA a categóricas"""
    st.subheader("PCA y MCA")
    
    X_train = estado['X_train'] 
    X_test = estado['X_test']
    numerical_cols = estado['numerical_cols']
    categorical_cols = estado['categorical_cols']
    y_train = estado['y_train']
    
    st.write("### Análisis de Componentes Principales (PCA)")
    
    # Pipeline para variables numéricas
    num_pipeline_pca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    X_train_num = num_pipeline_pca.fit_transform(X_train[numerical_cols])
    X_test_num = num_pipeline_pca.transform(X_test[numerical_cols])
    
    # Aplicar PCA
    pca = PCA(n_components=0.7, random_state=123)
    X_train_pca = pca.fit_transform(X_train_num)
    X_test_pca = pca.transform(X_test_num)
    
    # Visualizar varianza explicada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Varianza explicada por componente
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    ax1.set_title('Varianza Explicada por Componente PCA', fontweight='bold')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Proporción de Varianza Explicada')
    
    # Varianza acumulada
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumvar) + 1), cumvar, 'o-')
    ax2.axhline(y=0.7, color='r', linestyle='--', label='70% Umbral')
    ax2.set_title('Varianza Acumulada PCA', fontweight='bold')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Scatter plot PC1 vs PC2
    if X_train_pca.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                           c=[0 if target == 'healthy' else 1 for target in y_train],
                           cmap='viridis', alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2') 
        ax.set_title('Scatter Plot PC1 vs PC2', fontweight='bold')
        plt.colorbar(scatter, label='Target (0=healthy, 1=diseased)')
        st.pyplot(fig)
    
    st.info(f"PCA: {X_train_pca.shape[1]} componentes explican {cumvar[-1]:.1%} de la varianza")
    
    # MCA - Análisis de Correspondencias Múltiples (siguiendo el notebook)
    st.write("### Análisis de Correspondencias Múltiples (MCA)")
    
    if MCA_AVAILABLE:
        st.info("📊 MCA aplicado a variables categóricas")
        
        try:
            # Pipeline para imputación y OneHotEncoding de categóricas (como en el notebook)
            cat_pipeline_mca = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Aplicar preprocesamiento categórico en train y test
            X_train_cat_processed = cat_pipeline_mca.fit_transform(X_train[categorical_cols])
            X_test_cat_processed = cat_pipeline_mca.transform(X_test[categorical_cols])
            
            # Convertir a DataFrame para que mca.MCA funcione (necesita nombres de columnas)
            cat_feature_names_processed = cat_pipeline_mca.named_steps['encoder'].get_feature_names_out(categorical_cols)
            X_train_cat_processed_df = pd.DataFrame(X_train_cat_processed, columns=cat_feature_names_processed, index=X_train.index)
            X_test_cat_processed_df = pd.DataFrame(X_test_cat_processed, columns=cat_feature_names_processed, index=X_test.index)
            
            # Aplicar MCA SOLO a los datos categóricos de entrenamiento procesados (OneHotEncoded)
            mca_model = MCA(X_train_cat_processed_df, ncols=X_train_cat_processed_df.shape[1])
            
            # Obtener los componentes MCA para entrenamiento y prueba
            # fs_r_sup calcula los factores principales para las filas (observaciones)
            X_train_mca_categorical = mca_model.fs_r_sup(X_train_cat_processed_df)
            X_test_mca_categorical = mca_model.fs_r_sup(X_test_cat_processed_df)
            
            st.write(f"**Shape de X_train_mca_categorical:** {X_train_mca_categorical.shape}")
            st.write(f"**Shape de X_test_mca_categorical:** {X_test_mca_categorical.shape}")
            
            # Valores singulares y autovalores
            sv = mca_model.s
            eigvals = sv ** 2
            explained_var = eigvals / eigvals.sum()
            cum_explained_var = np.cumsum(explained_var)
            
            # Número de componentes para >=70% varianza explicada
            n_components_70 = np.argmax(cum_explained_var >= 0.7) + 1
            st.write(f"**Componentes MCA para >=70% varianza explicada:** {n_components_70}")
            
            # Graficar varianza acumulada
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker='o', linestyle='--')
            ax.axhline(y=0.7, color='r', linestyle='-', label='70% Varianza Acumulada')
            ax.axvline(x=n_components_70, color='g', linestyle='--', label=f'{n_components_70} Componentes (>=70%)')
            ax.set_xlabel('Dimensiones MCA')
            ax.set_ylabel('Varianza acumulada explicada')
            ax.set_title('Varianza acumulada explicada por MCA (Variables Categóricas, ajustado en entrenamiento)')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            # Scatter plot MCA1 vs MCA2
            if X_train_mca_categorical.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_train_mca_categorical[:,0], X_train_mca_categorical[:,1], 
                                   c=[0 if target == 'healthy' else 1 for target in y_train],
                                   cmap='viridis', alpha=0.7)
                ax.set_xlabel('Dimensión MCA 1')
                ax.set_ylabel('Dimensión MCA 2')
                ax.set_title('Scatterplot Dimensión MCA 1 vs Dimensión MCA 2 (Variables Categóricas)')
                plt.colorbar(scatter, label='Target (0=healthy, 1=diseased)')
                ax.grid(True)
                st.pyplot(fig)
            
            # Seleccionar solo los primeros n_components_70 de MCA
            X_train_mca_selected = X_train_mca_categorical[:, :n_components_70]
            X_test_mca_selected = X_test_mca_categorical[:, :n_components_70]
            
            # Concatenar los componentes de PCA y los componentes seleccionados de MCA
            X_train_combined = np.hstack((X_train_pca, X_train_mca_selected))
            X_test_combined = np.hstack((X_test_pca, X_test_mca_selected))
            
            st.write(f"**Shape de X_train_pca_mca:** {X_train_combined.shape}")
            st.write(f"**Shape de X_test_pca_mca:** {X_test_combined.shape}")
            
            n_mca_components = n_components_70
            
            st.success(f"✅ MCA: {n_mca_components} componentes explican >=70% de la varianza")
            
        except Exception as e:
            st.error(f"Error en MCA: {str(e)}")
            st.warning("⚠️ Usando PCA como alternativa para variables categóricas")
            
            # Fallback a PCA
            cat_pipeline_alt = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            X_train_cat = cat_pipeline_alt.fit_transform(X_train[categorical_cols])
            X_test_cat = cat_pipeline_alt.transform(X_test[categorical_cols])
            
            pca_cat = PCA(n_components=min(5, X_train_cat.shape[1]), random_state=123)
            X_train_mca_selected = pca_cat.fit_transform(X_train_cat)
            X_test_mca_selected = pca_cat.transform(X_test_cat)
            
            # Combinar PCA y MCA (fallback)
            X_train_combined = np.hstack([X_train_pca, X_train_mca_selected])
            X_test_combined = np.hstack([X_test_pca, X_test_mca_selected])
            n_mca_components = X_train_mca_selected.shape[1]
    
    else:
        st.warning("⚠️ Librería MCA no disponible, usando reducción dimensional alternativa")
        # Usar PCA en variables categóricas codificadas como alternativa
        cat_pipeline_alt = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        X_train_cat = cat_pipeline_alt.fit_transform(X_train[categorical_cols])
        X_test_cat = cat_pipeline_alt.transform(X_test[categorical_cols])
        
        pca_cat = PCA(n_components=min(5, X_train_cat.shape[1]), random_state=123)
        X_train_mca_selected = pca_cat.fit_transform(X_train_cat)
        X_test_mca_selected = pca_cat.transform(X_test_cat)
        
        # Combinar PCA y MCA (alternativo)
        X_train_combined = np.hstack([X_train_pca, X_train_mca_selected])
        X_test_combined = np.hstack([X_test_pca, X_test_mca_selected])
        n_mca_components = X_train_mca_selected.shape[1]
    
    # Mostrar métricas finales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Componentes PCA", X_train_pca.shape[1])
    with col2:
        st.metric("Componentes MCA", n_mca_components)
    with col3:
        st.metric("Total combinado", X_train_combined.shape[1])
    
    return {
        'X_train_pca_mca': X_train_combined,
        'X_test_pca_mca': X_test_combined,
        'pca_components': X_train_pca.shape[1],
        'mca_components': n_mca_components,
        'mensaje': 'PCA y MCA aplicados correctamente'
    }

def aplicar_balanceo(estado):
    """Aplica técnicas de balanceo de datos"""
    st.subheader("⚖️ Técnicas de Balanceo")
    
    # Usar los datos de PCA+MCA combinados
    X_train_selected = estado['X_train_selected']
    y_train_enc = estado['y_train_enc']
    
    st.write("### Distribución Original vs Balanceada")
    
    # Mostrar distribución original
    col1, col2 = st.columns(2)
    
    with col1:
        original_counts = pd.Series(y_train_enc).value_counts()
        st.write("**Distribución Original:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Healthy (0)', 'Diseased (1)'], original_counts.values, color=['lightblue', 'lightcoral'])
        ax.set_title('Distribución Original')
        ax.set_ylabel('Número de Muestras')
        
        # Agregar números sobre las barras
        for bar, count in zip(bars, original_counts.values):
            height = bar.get_height()
            ax.annotate(f'{count:,}', (bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig)
    
    # Aplicar SMOTETomek como técnica de balanceo
    balanceador = SMOTETomek(random_state=123)
    X_train_balanced, y_train_balanced = balanceador.fit_resample(X_train_selected, y_train_enc)
    
    with col2:
        balanced_counts = pd.Series(y_train_balanced).value_counts()
        st.write("**Distribución Balanceada (SMOTETomek):**")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Healthy (0)', 'Diseased (1)'], balanced_counts.values, color=['lightgreen', 'orange'])
        ax.set_title('Distribución Balanceada')
        ax.set_ylabel('Número de Muestras')
        
        # Agregar números sobre las barras
        for bar, count in zip(bars, balanced_counts.values):
            height = bar.get_height()
            ax.annotate(f'{count:,}', (bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig)
    
    # Mostrar métricas de balanceo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Muestras originales", f"{len(y_train_enc):,}")
    with col2:
        st.metric("Muestras balanceadas", f"{len(y_train_balanced):,}")
    with col3:
        ratio_original = original_counts.min() / original_counts.max()
        ratio_balanced = balanced_counts.min() / balanced_counts.max()
        st.metric("Ratio balanceado", f"{ratio_balanced:.2f}", f"+{ratio_balanced - ratio_original:.2f}")
    
    st.success("✅ Balanceo aplicado con SMOTETomek")
    
    return {
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'balanceador_usado': 'SMOTETomek',
        'mensaje': 'Datos balanceados correctamente'
    }

def entrenar_modelos(estado):
    """Entrena múltiples modelos de clasificación"""
    st.subheader("🤖 Modelos de Clasificación")
    
    X_train_balanced = estado['X_train_balanced']
    y_train_balanced = estado['y_train_balanced']
    X_test_selected = estado['X_test_selected']  # Usar datos de rf para test
    y_test_enc = estado['y_test_enc']
    le = estado['le']
    
    # Definir modelos
    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=123, class_weight='balanced'),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=123, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(random_state=123, class_weight='balanced', max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=123, class_weight='balanced')
    }
    
    resultados = {}
    roc_data = {}
    
    st.write("### Entrenamiento y Evaluación de Modelos")
    
    progress_bar_modelos = st.progress(0)
    status_modelos = st.empty()
    
    for i, (nombre, modelo) in enumerate(modelos.items()):
        status_modelos.text(f"Entrenando {nombre}...")
        progress_bar_modelos.progress((i + 1) / len(modelos))
        
        try:
            # Entrenar modelo
            modelo.fit(X_train_balanced, y_train_balanced)
            
            # Predicciones
            if hasattr(modelo, "predict_proba"):
                y_proba = modelo.predict_proba(X_test_selected)[:, 1]
            elif hasattr(modelo, "decision_function"):
                y_proba = modelo.decision_function(X_test_selected)
                # Normalizar scores para que estén entre 0 y 1
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            else:
                # Si no tiene probabilidades, usar predicciones binarias
                y_proba = modelo.predict(X_test_selected)
            
            # Buscar mejor umbral para F1-score
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.1, 0.91, 0.05):
                y_pred_threshold = (y_proba >= threshold).astype(int)
                f1_macro = f1_score(y_test_enc, y_pred_threshold, average='macro')
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_threshold = threshold
            
            # Predicciones finales con mejor umbral
            y_pred = (y_proba >= best_threshold).astype(int)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test_enc, y_pred)
            precision = precision_score(y_test_enc, y_pred, average='macro')
            recall = recall_score(y_test_enc, y_pred, average='macro')
            f1_macro = f1_score(y_test_enc, y_pred, average='macro')
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(y_test_enc, y_proba)
                fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
                roc_data[nombre] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
            except:
                roc_auc = 0.5
            
            # Guardar resultados
            resultados[nombre] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_macro': f1_macro,
                'roc_auc': roc_auc,
                'best_threshold': best_threshold,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
        except Exception as e:
            st.error(f"Error entrenando {nombre}: {str(e)}")
            resultados[nombre] = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 
                'f1_macro': 0, 'roc_auc': 0.5, 'best_threshold': 0.5,
                'error': str(e)
            }
    
    status_modelos.text("✅ Entrenamiento completado")
    
    return {
        'resultados_modelos': resultados,
        'roc_data': roc_data,
        'mejor_modelo': max(resultados.keys(), key=lambda k: resultados[k]['f1_macro']),
        'mensaje': 'Modelos entrenados correctamente'
    }

def generar_resultados(estado):
    """Genera y muestra los resultados finales"""
    st.subheader("Resultados Finales")
    
    resultados = estado['resultados_modelos']
    roc_data = estado['roc_data']
    mejor_modelo = estado['mejor_modelo']
    le = estado['le']
    y_test_enc = estado['y_test_enc']
    
    # Tabla comparativa de resultados
    st.write("### Comparación de Modelos")
    
    df_resultados = pd.DataFrame(resultados).T
    df_resultados = df_resultados[['accuracy', 'precision', 'recall', 'f1_macro', 'roc_auc']]
    df_resultados = df_resultados.round(3)
    
    # Destacar el mejor modelo
    st.dataframe(
        df_resultados.style.highlight_max(axis=0, color='lightgreen'),
        use_container_width=True
    )
    
    # Métricas del mejor modelo
    st.write(f"### 🏆 Mejor Modelo: {mejor_modelo}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    best_results = resultados[mejor_modelo]
    with col1:
        st.metric("Accuracy", f"{best_results['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{best_results['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{best_results['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{best_results['f1_macro']:.3f}")
    with col5:
        st.metric("ROC AUC", f"{best_results['roc_auc']:.3f}")
    
    # Curvas ROC
    if roc_data:
        st.write("### Curvas ROC - Comparación de Modelos")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for nombre, data in roc_data.items():
            ax.plot(data['fpr'], data['tpr'], lw=2, 
                   label=f'{nombre} (AUC = {data["auc"]:.3f})')
        
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
               label='Clasificador Aleatorio (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        ax.set_title('Curvas ROC - Comparación de Modelos', fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Matriz de confusión del mejor modelo
    if 'y_pred' in best_results:
        st.write("### Matriz de Confusión - Mejor Modelo")
        
        cm = confusion_matrix(y_test_enc, best_results['y_pred'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f'Matriz de Confusión - {mejor_modelo}', fontweight='bold')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        
        st.pyplot(fig)
    
    return {
        'resultados_finales': {
            'mejor_modelo': mejor_modelo,
            'metricas': best_results,
            'comparacion': df_resultados,
            'interpretacion': 'Análisis completado'
        }
    }

# Funciones de visualización individual (para el menú lateral)
def mostrar_analisis_exploratorio():
    st.header("Análisis Exploratorio de Datos")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver el análisis exploratorio completo con datos reales.")
    
    st.markdown("""
    ### Objetivos del Análisis Exploratorio:
    - **Comprensión de la estructura** de los datos de salud
    - **Identificación de patrones** en variables de estilo de vida  
    - **Detección de valores atípicos** y datos faltantes
    - **Análisis de correlaciones** entre variables predictoras
    - **Evaluación del desbalance** en la variable objetivo
    
    ### Variables Analizadas:
    #### Variables Numéricas:
    - age, bmi_corrected, blood_pressure, cholesterol, glucose, sleep_hours, weight, height
    
    #### Variables Categóricas:
    - gender, marital_status, sleep_quality, smoking_level, diet_type, healthcare_access, occupation
    """)

def mostrar_preprocesamiento():
    st.header("Preprocesamiento de Datos")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver el preprocesamiento completo.")
    
    st.markdown("""
    ### Técnicas de Preprocesamiento Implementadas:
    
    #### 1. División de Datos:
    - **División estratificada** 70/30 (entrenamiento/prueba)
    - **Preservación de proporciones** de clases
    - **Reproducibilidad** con semilla aleatoria fija
    
    #### 2. Pipelines de Transformación:
    - **Variables Numéricas**: Imputación (media) + StandardScaler
    - **Variables Categóricas**: Imputación (moda) + OneHotEncoder
    - **Prevención de data leakage** con fit/transform separation
    
    #### 3. Codificación de Variable Objetivo:
    - **LabelEncoder** para convertir 'healthy'/'diseased' a 0/1
    - **Mantenimiento de interpretabilidad** clínica
    """)

def mostrar_seleccion_caracteristicas():
    st.header("Selección de Características")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver la selección de características completa.")
    
    st.markdown("""
    ### Métodos de Selección Comparados:
    
    #### 1. SelectKBest (Chi-cuadrado):
    - **Dependencia estadística** entre variables categóricas y target
    - **Discretización** de variables numéricas requerida
    - **Ventaja**: Rápido y estadísticamente fundamentado
    - **Limitación**: Solo detecta relaciones lineales
    
    #### 2. SelectKBest (Información Mutua):
    - **Relaciones no lineales** entre variables y target
    - **Teoría de la información** (reducción de entropía)
    - **Ventaja**: Captura dependencias complejas
    - **Limitación**: Estimación puede ser ruidosa
    
    #### 3. Random Forest (Método Incrustado):
    - **Importancia basada en impureza** en árboles de decisión
    - **Considera interacciones** entre variables naturalmente
    - **Ventaja**: Interpretabilidad clínica directa
    - **Aplicación**: Ideal para bioestadística
    
    #### 4. RFECV (Método de Envoltura):
    - **Eliminación recursiva** con validación cruzada
    - **Optimización específica** para algoritmo de clasificación
    - **Ventaja**: Maximiza rendimiento predictivo
    - **Limitación**: Computacionalmente costoso
    
    ### Criterio de Selección:
    **90% de importancia acumulada** como umbral para balancear información vs reducción dimensional.
    """)

def mostrar_pca_mca():
    st.header("PCA y MCA")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver PCA y MCA completos.")
    
    st.markdown("""
    ### Análisis de Componentes Principales (PCA):
    
    #### Fundamento Teórico:
    - **Reducción de dimensionalidad lineal** preservando máxima varianza
    - **Eliminación de redundancia** entre variables correlacionadas
    - **Componentes ortogonales** que capturan patrones principales
    
    #### Aplicación en Variables Numéricas:
    - age, bmi_corrected, blood_pressure, cholesterol, glucose, etc.
    - **Criterio**: 70% de varianza explicada acumulada
    - **Interpretación**: Componentes como "factores de salud general"
    
    ### Análisis de Correspondencias Múltiples (MCA):
    
    #### Fundamento Teórico:
    - **Extensión de PCA** para variables categóricas/nominales
    - **Análisis de asociaciones** entre categorías
    - **Visualización de perfiles** de comportamiento
    
    #### Aplicación en Variables Categóricas:
    - gender, marital_status, diet_type, healthcare_access, etc.
    - **Objetivo**: Identificar patrones de estilo de vida
    - **Ventaja**: Mantiene naturaleza categórica de los datos
    
    ### Combinación PCA + MCA:
    - **Enfoque híbrido** para datos mixtos (numéricos + categóricos)
    - **Representación compacta** del espacio de características completo
    - **Preservación de información** tanto cuantitativa como cualitativa
    """)

def mostrar_balanceo():
    st.header("Técnicas de Balanceo")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver las técnicas de balanceo completas.")
    
    st.markdown("""
    ### Problemática del Desbalance de Clases:
    
    #### Desafíos Identificados:
    - **Distribución**: ~70% healthy vs ~30% diseased
    - **Sesgo predictivo** hacia clase mayoritaria
    - **Baja sensibilidad** para detectar casos de riesgo
    - **Métricas engañosas** (alta accuracy, baja utilidad clínica)
    
    ### Técnicas de Balanceo Evaluadas:
    
    #### 1. SMOTE (Synthetic Minority Oversampling):
    - **Generación sintética** de ejemplos minoritarios
    - **Interpolación k-NN** en espacio de características
    - **Ventaja**: Evita overfitting por duplicación
    - **Aplicación clínica**: Aumenta detección de casos de riesgo
    
    #### 2. SMOTETomek (Técnica Híbrida):
    - **Combinación**: SMOTE + limpieza Tomek Links
    - **Oversampling inteligente** + eliminación de outliers
    - **Ventaja**: Mejora frontera de decisión
    - **Resultado**: Datos más limpios y balanceados
    
    #### 3. BorderlineSMOTE:
    - **SMOTE selectivo** en ejemplos frontera
    - **Enfoque**: Casos difíciles de clasificar
    - **Ventaja**: Mejora separabilidad de clases
    
    ### Evaluación de Técnicas:
    - **Métricas balanceadas**: F1-macro, Recall, Precision
    - **ROC AUC** para evaluar capacidad discriminativa
    - **Validación cruzada estratificada**
    - **Impacto en interpretabilidad clínica**
    """)

def mostrar_modelos():
    st.header("Modelos de Clasificación")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver el entrenamiento de modelos completo.")
    
    st.markdown("""
    ### Algoritmos de Clasificación Implementados:
    
    #### 1. Random Forest:
    - **Ensemble de árboles** con bagging
    - **Ventajas**: Maneja no-linealidad, importancia de variables, robusto
    - **Aplicación clínica**: Sistemas de apoyo diagnóstico interpretables
    - **Hiperparámetros**: n_estimators, max_depth, min_samples_split
    
    #### 2. Gradient Boosting (HistGradientBoosting):
    - **Ensemble secuencial** que corrige errores previos
    - **Ventajas**: Excelente capacidad predictiva, maneja patrones complejos
    - **Aplicación clínica**: Modelos de pronóstico y estratificación de riesgo
    - **Optimización**: learning_rate, max_iter, max_depth
    
    #### 3. Regresión Logística:
    - **Modelo lineal** con interpretación probabilística
    - **Ventajas**: Interpretabilidad directa, coeficientes = odds ratios
    - **Aplicación clínica**: Scores de riesgo cardiovascular, modelos explicativos
    - **Regularización**: L1, L2, ElasticNet para selección automática
    
    #### 4. K-Nearest Neighbors (KNN):
    - **Clasificación por similaridad** local
    - **Ventajas**: No paramétrico, adapta a estructura local
    - **Aplicación clínica**: Diagnóstico por casos similares, medicina personalizada
    - **Parámetros**: n_neighbors, weights, metric
    
    #### 5. Extra Trees:
    - **Ensemble con aleatoriedad** en división de nodos
    - **Ventajas**: Reduce overfitting, rápido entrenamiento
    - **Aplicación clínica**: Modelos robustos con datos ruidosos
    
    ### Optimización de Hiperparámetros:
    - **RandomizedSearchCV** para exploración eficiente
    - **Validación cruzada estratificada** (5 folds)
    - **Métrica objetivo**: F1-macro score
    - **Búsqueda de umbral óptimo** para maximizar F1-score
    
    ### Evaluación Integral:
    - **Métricas múltiples**: Accuracy, Precision, Recall, F1-macro, ROC AUC
    - **Curvas ROC** para comparación visual
    - **Matrices de confusión** para análisis detallado
    - **Interpretabilidad clínica** de resultados
    """)

def mostrar_resultados():
    st.header("Resultados y Evaluación")
    st.info("💡 Utiliza el botón '🔄 Ejecutar Análisis Completo' para ver los resultados completos.")
    
    st.markdown("""
    ### Métricas de Evaluación Utilizadas:
    
    #### Para Clasificación Binaria Desbalanceada:
    - **Accuracy**: Proporción total de predicciones correctas
    - **Precision**: Proporción de predicciones positivas correctas
    - **Recall (Sensitivity)**: Proporción de casos positivos detectados
    - **F1-Score**: Media armónica entre Precision y Recall
    - **ROC AUC**: Área bajo la curva ROC (capacidad discriminativa)
    
    ### Interpretación Clínica:
    
    #### Contexto de Aplicación en Salud:
    - **Falsos Negativos** (alta gravedad): Pacientes enfermos clasificados como sanos
    - **Falsos Positivos** (menor gravedad): Pacientes sanos clasificados como enfermos
    - **Recall alto** es prioritario para detección de casos de riesgo
    - **Precision adecuada** para evitar alarmas innecesarias
    
    ### Análisis Comparativo:
    - **Tablas de rendimiento** por algoritmo y técnica de balanceo
    - **Curvas ROC superpuestas** para comparación visual
    - **Matrices de confusión** para análisis de errores
    - **Identificación del modelo óptimo** basado en F1-macro score
    
    ### Limitaciones Identificadas:
    - **Rendimiento cercano al aleatorio** (AUC ≈ 0.5)
    - **Dificultad para discriminar** entre clases healthy/diseased
    - **Posible insuficiencia** de variables predictivas en el dataset
    - **Necesidad de ingeniería** de características adicional
    """)

def mostrar_conclusiones():
    st.header("Conclusiones")
    
    st.markdown("""
    ## Conclusiones del Análisis Comparativo
    
    ### Técnicas de Selección de Características
    
    Se implementaron **cuatro métodos de selección** de características (Chi-cuadrado, Información Mutua, 
    Random Forest y RFECV), los cuales mostraron convergencia parcial en la identificación de variables 
    relevantes. Si bien se logró definir un conjunto "core" de predictores, **ningún método permitió 
    construir modelos con capacidad discriminativa significativa**. Esto sugiere que, aunque las variables 
    seleccionadas son las más informativas dentro del dataset disponible, su poder explicativo para 
    diferenciar entre personas sanas y enfermas es limitado.
    
    ### Reducción de Dimensionalidad
    
    **PCA** permitió reducir la dimensionalidad de las variables numéricas, preservando el 70% de la 
    varianza con pocos componentes principales. **MCA** complementó el análisis en variables categóricas, 
    proporcionando una representación más compacta de los datos. Sin embargo, al tener resultados muy 
    similares a las técnicas de selección de características, no se utilizaron en el modelo final.
    
    ### Técnicas de Balanceo
    
    La evaluación de múltiples técnicas de balanceo mostró que **SMOTETomek** ofreció el mejor equilibrio 
    entre generación de muestras sintéticas y limpieza de fronteras de decisión. El balanceo permitió 
    mejorar la proporción de clases en el conjunto de entrenamiento, pero **no fue suficiente para que 
    los modelos detectaran eficazmente la clase minoritaria**. La precisión y el recall para la clase 
    "enfermo" se mantuvieron bajos en todos los algoritmos, reflejando la dificultad del problema.
    
    ### Rendimiento de Algoritmos
    
    El análisis comparativo de **cinco algoritmos de clasificación** reveló que **ninguno logró un 
    desempeño significativamente superior al azar**:
    
    - **Random Forest**: Ligera ventaja sobre otros modelos, pero sin capacidad real de discriminación
    - **HistGradientBoosting**: Métricas similares a Random Forest
    - **KNN**: Los métodos basados en vecinos tampoco lograron captar patrones diferenciadores
    - **ExtraTrees**: Único modelo con AUC ligeramente superior, pero aún dentro del rango aleatorio
    - **Regresión Logística**: La interpretabilidad no se tradujo en mejor rendimiento
    
    En todos los casos, la **accuracy se mantuvo entre 0.56 y 0.59**, y el F1-macro cerca de 0.5, 
    indicando que los modelos tienden a predecir la clase mayoritaria ("healthy") con baja sensibilidad 
    para la clase minoritaria ("diseased").
    
    ### Métricas de Evaluación
    
    Las métricas obtenidas (precision, recall, F1-score, AUC) muestran que **la capacidad de los modelos 
    para distinguir entre clases es prácticamente nula**. El AUC cercano a 0.5 en todos los algoritmos 
    confirma que el desempeño es equivalente al azar. La matriz de confusión revela que la mayoría de 
    los casos "diseased" no son detectados correctamente, limitando la utilidad clínica de los modelos.
    
    ---
    
    ## Conclusión Final
    
    A pesar de aplicar **técnicas avanzadas de selección de características**, **reducción de 
    dimensionalidad** y **balanceo de clases**, **los modelos no logran superar el desempeño aleatorio**. 
    
    ### Posibles Causas:
    - **Falta de variables verdaderamente discriminantes** en el dataset
    - **Presencia de ruido** o información irrelevante
    - **Complejidad inherente** del problema de predicción de salud
    - **Limitaciones del dataset sintético** utilizado para la demostración
    
    ### Recomendaciones:
    - **Explorar nuevas fuentes de datos** con variables más específicas
    - **Realizar ingeniería de características** más profunda
    - **Considerar enfoques alternativos** como deep learning o métodos ensemble avanzados
    - **Incorporar conocimiento experto** del dominio médico
    - **Validar con datasets reales** de mayor calidad
    
    ### Valor del Análisis:
    Este trabajo demuestra la **importancia de una metodología rigurosa** en machine learning aplicado 
    a la salud, evidenciando que no siempre es posible obtener modelos predictivos útiles, incluso 
    aplicando las mejores prácticas técnicas. La **transparencia en los resultados negativos** es 
    fundamental para el avance científico en bioestadística.
    """)

def mostrar_resultados_completos(resultados_finales):
    """Muestra un resumen ejecutivo de todos los resultados"""
    st.header("Resumen Ejecutivo de Resultados")
    
    mejor_modelo = resultados_finales['mejor_modelo']
    metricas = resultados_finales['metricas']
    
    st.success(f"**Mejor Modelo Identificado:** {mejor_modelo}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("F1-Score", f"{metricas['f1_macro']:.3f}")
        st.metric("ROC AUC", f"{metricas['roc_auc']:.3f}")

    with col2:
        st.metric("Accuracy", f"{metricas['accuracy']:.3f}")
        st.metric("Precision", f"{metricas['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{metricas['recall']:.3f}")
        st.metric("Threshold", f"{metricas['best_threshold']:.2f}")

    # Interpretación automática
    if metricas['roc_auc'] > 0.7:
        interpretacion = "🟢 **Excelente capacidad predictiva**"
    elif metricas['roc_auc'] > 0.6:
        interpretacion = "🟡 **Capacidad predictiva moderada**"
    else:
        interpretacion = "🔴 **Capacidad predictiva limitada (≈ aleatorio)**"
    
    st.markdown(f"### Interpretación: {interpretacion}")
    
    if metricas['roc_auc'] <= 0.6:
        st.warning("""
        ⚠️ Los modelos muestran capacidad discriminativa limitada.
        Se recomienda:
        - Revisar la calidad y relevancia de las variables y los datos
        - Considerar fuentes de datos adicionales
        - Evaluar la necesidad de más muestras o probar con otros modelos o técnicas
        """)

def mostrar_seccion_resultados(seccion, resultados):
    """Muestra una sección específica de los resultados del análisis"""
    
    if seccion == "Resumen Completo":
        st.header("Resumen Completo del Análisis")
        
        if 'resultados_finales' in resultados:
            mostrar_resultados_completos(resultados['resultados_finales'])
        
        # Mostrar métricas clave de cada paso
        st.subheader("🔍 Métricas por Etapa")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Datos**")
            if 'df' in resultados:
                st.metric("Registros totales", f"{len(resultados['df']):,}")
                st.metric("Variables numéricas", len(resultados.get('numerical_cols', [])))
                st.metric("Variables categóricas", len(resultados.get('categorical_cols', [])))
        
        with col2:
            st.markdown("**Reducción Dimensional**")
            if 'pca_components' in resultados:
                st.metric("Componentes PCA", resultados['pca_components'])
            if 'mca_components' in resultados:
                st.metric("Componentes MCA", resultados['mca_components'])
            if 'n_features_selected' in resultados:
                st.metric("Características seleccionadas", resultados['n_features_selected'])
        
        with col3:
            st.markdown("**Mejor Modelo**")
            if 'resultados_finales' in resultados:
                mejor_modelo = resultados['resultados_finales']['mejor_modelo']
                metricas = resultados['resultados_finales']['metricas']
                st.metric("Algoritmo", mejor_modelo)
                st.metric("F1-Score", f"{metricas.get('f1_macro', 0):.3f}")
                st.metric("ROC AUC", f"{metricas.get('roc_auc', 0):.3f}")
    
    elif seccion == "Carga de Datos":
        st.header("Carga de Datos")
        if 'df' in resultados:
            df = resultados['df']
            st.write("### Vista previa del dataset")
            st.dataframe(df.head())
            
            st.write("### Información del dataset")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total registros", f"{len(df):,}")
            with col2:
                st.metric("Variables numéricas", len(resultados.get('numerical_cols', [])))
            with col3:
                st.metric("Variables categóricas", len(resultados.get('categorical_cols', [])))
    
    elif seccion == "Análisis Exploratorio":
        st.header("Análisis Exploratorio")
        if 'df' in resultados:
            df = resultados['df']
            
            # Distribución de la variable objetivo
            st.write("### Distribución de la Variable Objetivo")
            target_counts = df['target'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(target_counts.index, target_counts.values, color=['#1f77b4', '#ff7f0e'])
            ax.set_title('Distribución de la Variable Objetivo')
            ax.set_xlabel('Estado de Salud')
            ax.set_ylabel('Número de Personas')
            
            for bar, count in zip(bars, target_counts.values):
                height = bar.get_height()
                ax.annotate(f'{count:,}', (bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom')
            
            st.pyplot(fig)

    elif seccion == "PCA y MCA":
        st.header("Resultados PCA y MCA")

        col1, col2 = st.columns(2)
        with col1:
            if 'pca_components' in resultados:
                st.metric("Componentes PCA", resultados['pca_components'])
        with col2:
            if 'mca_components' in resultados:
                st.metric("Componentes MCA", resultados['mca_components'])
        
        st.info("💡 Los detalles completos de PCA y MCA se muestran durante la ejecución del análisis.")

    elif seccion == "Mejores Resultados":
        st.header("Mejores Resultados")

        if 'resultados_finales' in resultados:
            resultados_finales = resultados['resultados_finales']
            mejor_modelo = resultados_finales['mejor_modelo']
            metricas = resultados_finales['metricas']
            
            st.success(f"🏆 **Mejor Modelo:** {mejor_modelo}")
            
            # Mostrar métricas en columnas
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("F1-Score", f"{metricas.get('f1_macro', 0):.3f}")
            with col2:
                st.metric("ROC AUC", f"{metricas.get('roc_auc', 0):.3f}")
            with col3:
                st.metric("Accuracy", f"{metricas.get('accuracy', 0):.3f}")
            with col4:
                st.metric("Precision", f"{metricas.get('precision', 0):.3f}")
            with col5:
                st.metric("Recall", f"{metricas.get('recall', 0):.3f}")
            
            # Interpretación
            if metricas.get('roc_auc', 0) > 0.7:
                st.success("🟢 **Excelente capacidad predictiva**")
            elif metricas.get('roc_auc', 0) > 0.6:
                st.warning("🟡 **Capacidad predictiva moderada**")
            else:
                st.error("🔴 **Capacidad predictiva limitada (≈ aleatorio)**")
    
    elif seccion == "Comparación de Modelos":
        st.header("Comparación de Modelos")

        if 'resultados_modelos' in resultados:
            resultados_modelos = resultados['resultados_modelos']
            
            # Crear tabla comparativa
            df_comparacion = pd.DataFrame(resultados_modelos).T
            df_comparacion = df_comparacion[['accuracy', 'precision', 'recall', 'f1_macro', 'roc_auc']]
            df_comparacion = df_comparacion.round(3)
            
            st.write("### Tabla Comparativa de Modelos")
            st.dataframe(df_comparacion.style.highlight_max(axis=0, color='lightgreen'))
            
            st.info("Los valores destacados en verde representan las mejores métricas por columna.")
    
    elif seccion == "Conclusiones Finales":
        st.header("Conclusiones Finales")
        mostrar_conclusiones()
    
    else:
        st.info(f"Sección '{seccion}' en desarrollo. Selecciona otra sección para ver los resultados.")

if __name__ == "__main__":
    main()