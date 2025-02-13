# 📰 Fake News Detection - Detección de Noticias Falsas en Contexto Político

## 📌 Descripción
Este proyecto desarrolla un sistema automatizado para la detección de **noticias falsas** en el ámbito político, con especial énfasis en **procesos electorales en España**. Utiliza **Procesamiento del Lenguaje Natural (PLN)** y **aprendizaje automático** para clasificar noticias como verdaderas o falsas, y se implementa en un servidor web accesible.

## 🎯 Objetivos
- Recopilar y procesar un conjunto de datos de noticias políticas verdaderas y falsas.
- Desarrollar y entrenar un modelo basado en **BERT** con los datos obtenidos y técnicas de PLN.
- Implementar un sistema de captura y clasificación de noticias en tiempo real.
- Desplegar una **interfaz web** para consultar análisis y clasificaciones.

## 🔧 Tecnologías y Herramientas
- **Lenguajes:** Python
- **Bibliotecas:** Pandas, Scikit-learn, Transformers (Hugging Face), Selenium, BeautifulSoup
- **Modelos utilizados:**
  - **BERT** (para generación de resúmenes y títulos)
  - **Random Forest y SVM** (clasificación inicial)
  - **DuckDuckGo y BERT** para detección de contexto
  - **SaBERT** (modelo especializado en noticias falsas en español)
- **Infraestructura:** Google Colab, servidores locales

## 📂 Estructura del Proyecto
```
📂 Fake-News-Detection
│── 📁 pasar_a_tweet          # Notebooks para pasar a formato tweet las noticias
│── 📁 entrenamiento_bert     # Entrenamiento del modelo con los datos obtenidos
│── 📁 preprocesamiento       # Procesamiento con Pandas y NLTK de los datos obtenidos
│── 📁 scrap_verdaderas       # Notebooks para obtener las noticias verdaderas mediante web scrapping
│── 📁 scrap_falsas           # Notebook para la extracción de noticias falsas
│── 📁 datasets_finales       # Datasets obtenidos mediante la extracción web
│── 📁 fastApi   
│── ├── 📁 data               # Noticias extraídas mediante el servidor web para el análisis 
│── ├── ├── fakes1000          # Dataset que sirve para probar los modelos del servidor
│── ├── 📁 static             # Elementos visuales de la página  
│── ├── 📁 templates          # Elementos visuales de la página  
│── ├── classifier             # Clasificador de noticias diarias
│── ├── main            
│── ├── modelo                # Modelos usados para la clasificación
│── ├── scrapper
│── dataset_unificado         # Dataset de noticias verdaderas y falsas
│── dataset_unificado_pro     # Dataset de noticias verdaderas y falsas procesado con NLTK
│── requirements.txt          # Dependencias necesarias
```

## 🚀 Instalación y Uso
1. **Clona este repositorio:**
   ```bash
   git clone https://github.com/tuusuario/Fake-News-Detection.git
   cd Fake-News-Detection
   ```
2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Inicia la API para clasificación en tiempo real:**
   ```bash
   python src/api/main.py
   ```
4. **Accede a la interfaz web en el navegador:**
   ```
   http://localhost:5000
   ```

## 📊 Resultados y Métricas
- Precisión del modelo: **95% con BERT**
- Evaluación con métricas estándar (F1-score, Recall, Precision)
- Comparación entre modelos tradicionales y avanzados

## 📌 Contribuciones
¡Las contribuciones son bienvenidas! Si deseas mejorar el código, abre un **issue** o envía un **pull request**.
El proyecto está en fase de construción por lo que se incorporará un diagrama y guía de uso detallado más adelante.

## 📜 Licencia
Este proyecto está bajo la licencia **MIT**.
