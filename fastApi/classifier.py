import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from textblob import TextBlob
from transformers import pipeline
import logging


LABELS = {
    1: "Seguramente verdadera",
    2: "Posiblemente verdadera",
    3: "Inconcluso",
    4: "Posiblemente falsa",
    5: "Seguramente falsa",
}

STATIC_DATASET_PATHS = [
    "data/news_scrapped_static/elpais_news.csv",
    "data/news_scrapped_static/elmundo_news.csv",
    "data/news_scrapped_static/eldiario_news.csv",
    "data/news_scrapped_static/larazon_news.csv",
    "data/news_scrapped_static/infolibre_news.csv",
]

SCRAPED_DATASET_PATHS = [
    "data/news_scrapped/elpais_news.csv",
    "data/news_scrapped/elmundo_news.csv",
    "data/news_scrapped/eldiario_news.csv",
    "data/news_scrapped/larazon_news.csv",
    "data/news_scrapped/theobjective_news.csv",
    "data/news_scrapped/infolibre_news.csv",
]

# Load BERT model and tokenizer
BERT_MODEL_DIR = "models/bert"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR, num_labels=5)

# Load summarization model
summarizer_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", use_fast=False)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

# Load SaBERT-Spanish-Fake-News model
sabert_tokenizer = BertTokenizer.from_pretrained("VerificadoProfesional/SaBERT-Spanish-Fake-News")
sabert_model = BertForSequenceClassification.from_pretrained("VerificadoProfesional/SaBERT-Spanish-Fake-News")


# Function to extract stylistic features
def extract_stylistic_features(text):
    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
    length = len(text.split())
    return np.array([polarity, subjectivity, length])


# Normalize stylistic features
def normalize_features(features, num_labels=5):
    probabilities = np.zeros(num_labels)
    if features.sum() > 0:
        probabilities[:len(features)] = features / features.sum()
    return probabilities


# Predict with BERT
def predict_with_bert(texts):
    if not isinstance(texts, list):
        texts = [texts]
    tokens = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    outputs = bert_model(**tokens)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()
    return probabilities


# Predict with SaBERT-Spanish-Fake-News
def predict_with_sabert(text):
    inputs = sabert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = sabert_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # Map SaBERT probabilities to 5-class format
    extended_probs = [0] * 5
    extended_probs[0] = probabilities[0]  # False -> Seguramente falsa
    extended_probs[4] = probabilities[1]  # True -> Seguramente verdadera
    return extended_probs


# Search with DuckDuckGo
def search_duckduckgo(query, max_results=10):

    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        return [result.get('body', '') for result in results if 'body' in result]
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return []


# Summarize search results
def summarize_search_results(query, results):
    concatenated_results = "\n".join(results)
    inputs = summarizer_tokenizer(
        concatenated_results, return_tensors="pt", truncation=True, max_length=512
    )
    outputs = summarizer_model.generate(inputs.input_ids, max_length=200, min_length=50, length_penalty=2.0,
                                        num_beams=4)
    summary = summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Combine predictions for basic classification
def basic_prediction(text):
    bert_probs = predict_with_bert([text])[0]
    stylistic_features = extract_stylistic_features(text)
    stylistic_probs = normalize_features(stylistic_features, num_labels=len(bert_probs))

    weights = {"bert": 0.8, "stylistic": 0.2}
    combined_probs = (
            weights["bert"] * bert_probs +
            weights["stylistic"] * stylistic_probs
    )
    final_label = np.argmax(combined_probs) + 1
    return final_label, combined_probs


# Combine predictions for advanced classification
def advanced_prediction(title):


    bert_probs = predict_with_bert([title])[0]
    evidence = search_duckduckgo(title, max_results=10)
    if evidence:
        summary = summarize_search_results(title, evidence)
        summary_probs = predict_with_bert([summary])[0]
    else:
        summary_probs = bert_probs

    sabert_probs = predict_with_sabert(title)

    weights = {"bert": 0.3, "sabert": 0.3, "duckduckgo": 0.4}
    combined_probs = (
            weights["bert"] * bert_probs +
            weights["sabert"] * np.array(sabert_probs) +
            weights["duckduckgo"] * summary_probs
    )

    final_label = np.argmax(combined_probs) + 1
    return final_label, combined_probs


# Classify news
def classify_news(text, mode="basic"):
    if mode == "basic":
        return basic_prediction(text)
    elif mode == "advanced":
        return advanced_prediction(text)
    else:
        raise ValueError("Invalid classification mode. Choose 'basic' or 'advanced'.")


# Mapea las clasificaciones numéricas a etiquetas textuales
def map_classification_label(class_num):
    return LABELS.get(class_num, "Desconocido")


# Actualizar la función classify_news_combined
def classify_news_combined(use_static_data=False, mode="basic"):

    dataset_paths = STATIC_DATASET_PATHS if use_static_data else SCRAPED_DATASET_PATHS
    combined_data = []

    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            combined_data.append(df)
        else:
            print(f"File not found: {dataset_path}")

    if not combined_data:
        print("No datasets found to process.")
        return pd.DataFrame()

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.dropna(subset=["title", "url", "content"], inplace=True)

    combined_df["Clasificación"] = combined_df["title"].apply(lambda x: classify_news(x, mode=mode)[0])
    combined_df["Clasificación"] = combined_df["Clasificación"].map(LABELS)
    combined_df["Contenido"] = combined_df["content"].str[:50] + "..."

    combined_df = combined_df.rename(
        columns={
            "title": "Título",
            "content": "Contenido",
            "url": "URL",
            "source": "Fuente",
            "Clasificación": "Clasificación"
        }
    )

    return combined_df


if __name__ == "__main__":
    classified_news = classify_news_combined(use_static_data=True, mode="advanced")
    if not classified_news.empty:
        classified_news.to_csv("classified_news.csv", index=False)
        print("Clasificación completada. Resultados guardados en classified_news.csv.")
