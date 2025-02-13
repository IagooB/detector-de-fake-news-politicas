#%%
import pandas as pd
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import scipy.sparse

#%%
dataset_unificado = pd.read_csv("data/dataset_unificado.csv")
#%%
# Configuración de stopwords y lematización
stop_words = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()


# Preprocesamiento avanzado del texto
def procesar_texto_complejo(texto):
    # Eliminar URLs
    texto = re.sub(r'http\S+', '', texto)
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', texto)
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar stopwords y lematizar
    palabras = texto.split()
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras)

# Aplicar preprocesamiento al contenido y título
dataset_unificado["Contenido"] = dataset_unificado["Contenido"].apply(procesar_texto_complejo)
dataset_unificado["Título"] = dataset_unificado["Título"].apply(procesar_texto_complejo)

# Guardar dataset preprocesado
dataset_unificado.to_csv("data/dataset_unificado_preprocesado.csv", index=False)

#%%

# TF-IDF para representar el texto
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(dataset_unificado["Contenido"] + " " + dataset_unificado["Título"])

# Guardar el TF-IDF para futuros usos
scipy.sparse.save_npz("X_tfidf.npz", X_tfidf)

# División de los datos para entrenamiento, validación y prueba
y = dataset_unificado["etiqueta"]
X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_text = dataset_unificado["Contenido"] + " " + dataset_unificado["Título"]
y = dataset_unificado["etiqueta"]
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42)

#%%

# Guardar datos procesados
scipy.sparse.save_npz("X_train_tfidf.npz", X_train)
scipy.sparse.save_npz("X_val_tfidf.npz", X_val)
scipy.sparse.save_npz("X_test_tfidf.npz", X_test)
y_train.to_csv("y_train_final.csv", index=False)
y_val.to_csv("y_val_final.csv", index=False)
y_test.to_csv("y_test_final.csv", index=False)

print("Preprocesamiento avanzado completado y datos guardados.")
#%% md
# # Modelar
#%%
X_train = scipy.sparse.load_npz("data/procesed/X_train_tfidf.npz")
X_val = scipy.sparse.load_npz("data/procesed/X_val_tfidf.npz")
X_test = scipy.sparse.load_npz("data/procesed/X_test_tfidf.npz")
y_train = pd.read_csv("data/procesed/y_train_final.csv").values.ravel()
y_val = pd.read_csv("data/procesed/y_val_final.csv").values.ravel()
y_test = pd.read_csv("data/procesed/y_test_final.csv").values.ravel()

#%%
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "Support Vector Machine": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }
}

best_model = None
best_score = 0
best_name = None

#%%

for name, model in models.items():
    print(f"Optimizando modelo: {name}")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Mejor puntuación para {name}: {grid.best_score_}")
    print(f"Mejores parámetros: {grid.best_params_}")

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = name

#%%

print(f"Mejor modelo: {best_name}")
print(f"Entrenando en conjunto de validación...")
best_model.fit(X_train, y_train)
y_val_pred = best_model.predict(X_val)
print("Evaluación en conjunto de validación:")
print(classification_report(y_val, y_val_pred))
print("Accuracy en validación:", accuracy_score(y_val, y_val_pred))

# Evaluar en conjunto de prueba
y_test_pred = best_model.predict(X_test)
print("Evaluación en conjunto de prueba:")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Accuracy en prueba:", accuracy_score(y_test, y_test_pred))

#%%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib


# Pipeline que incluye TF-IDF
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', SVC(probability=True, random_state=42))
])

# Entrenar el pipeline con el texto plano
pipeline.fit(X_train_text, y_train)

# Guardar el pipeline entrenado
joblib.dump(pipeline, 'models/model.pkl')

#%%
X_train
#%%
