# Análisis Comparativo de Técnicas de Reducción de Dimensionalidad - Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

Una aplicación web interactiva desarrollada con Streamlit que presenta un análisis exhaustivo de técnicas de reducción de dimensionalidad aplicadas a la predicción del estado de salud usando datos de estilo de vida.

## Descripción del Proyecto

### **Trabajo Final - Machine Learning en Bioestadística**
**Autores:** David Zabala y Kevin Suarez  
**Fecha:** Agosto 2025

Este proyecto compara metodológicamente tres enfoques principales:
- **Análisis de Componentes Principales (PCA)**
- **Análisis de Correspondencias Múltiples (MCA)** 
- **Técnicas de selección de características**

## Características Principales

### **Análisis Completo Automatizado**
- Botón de ejecución completa que ejecuta todo el pipeline paso a paso
- Análisis exploratorio con visualizaciones interactivas
- 4 técnicas de selección de características comparadas
- PCA y MCA para reducción dimensional
- 8 técnicas de balanceo evaluadas
- 5 algoritmos de clasificación optimizados
- Evaluación integral con múltiples métricas

### **Navegación Intuitiva**
- Interfaz sidebar para explorar secciones específicas
- Visualizaciones en tiempo real
- Interpretación automática de resultados
- Diseño responsive

## Características

- **Interfaz interactiva**: Navegación intuitiva por diferentes secciones del análisis
- **Análisis completo**: Desde exploración de datos hasta evaluación de modelos
- **Visualizaciones**: Gráficos interactivos y estáticos
- **Ejecución paso a paso**: Botón para ejecutar todo el análisis de forma secuencial
- **Responsive**: Adaptado para diferentes tamaños de pantalla

## Instalación y Uso

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación local

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/proyecto-ml-bioestadistica.git
   cd proyecto-ml-bioestadistica
   ```

2. **Crea un entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecuta la aplicación**:
   ```bash
   streamlit run app.py
   ```

5. **Abre tu navegador** en `http://localhost:8501`

### Deployment en Streamlit Cloud

1. **Sube tu código a GitHub**
2. **Visita** [Streamlit Cloud](https://streamlit.io/cloud)
3. **Conecta tu repositorio** y selecciona la rama principal
4. **Especifica** `app.py` como archivo principal
5. **Deploy** automático

## Estructura del Proyecto

```
proyecto-ml-bioestadistica/
├── app.py                 # Aplicación principal de Streamlit
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
├── data/                 # Datos del proyecto (si los hay)
│   └── dataset.csv
└── .github/
    └── copilot-instructions.md
```

## Uso de la Aplicación

### Navegación Principal

La aplicación cuenta con un menú lateral que permite navegar por las siguientes secciones:

- **Introducción**: Objetivos y descripción del proyecto
- **Análisis Exploratorio**: Exploración inicial de los datos
- **Preprocesamiento**: Limpieza y preparación de datos
- **Modelado**: Implementación de modelos de ML
- **Resultados**: Evaluación y métricas de los modelos
- **Conclusiones**: Resumen y conclusiones finales

### Ejecución Completa

Utiliza el botón "🚀 Iniciar Análisis Completo" para ejecutar todo el pipeline de análisis de forma secuencial, incluyendo:

1. Carga y validación de datos
2. Análisis exploratorio automatizado
3. Preprocesamiento de variables
4. Entrenamiento de modelos
5. Evaluación y comparación
6. Generación de visualizaciones

## Funcionalidades

- ✅ Carga automática de datasets
- ✅ Análisis exploratorio interactivo
- ✅ Preprocesamiento de datos
- ✅ Múltiples algoritmos de ML
- ✅ Evaluación comparativa de modelos
- ✅ Visualizaciones interactivas
- ✅ Exportación de resultados
- ✅ Responsive design

## Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Algoritmos de machine learning
- **Matplotlib/Seaborn**: Visualizaciones estáticas
- **Plotly**: Visualizaciones interactivas

## Datasets Soportados

La aplicación está diseñada para trabajar con datasets biomédicos que incluyan:
- Variables numéricas y categóricas
- Variables objetivo binarias o multiclase
- Datos clínicos y de laboratorio

## Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autores

**Kevin Suárez** - Maestría en Estadística Aplicada y Ciencia de Datos
**David Zabala** - Maestría en Estadística Aplicada y Ciencia de Datos
