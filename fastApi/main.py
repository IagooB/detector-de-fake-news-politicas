from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from classifier import classify_news_combined
from scrapper import (
    scrape_elpais,
    scrape_el_mundo,
    scrape_el_diario,
    scrape_la_razon,
    scrape_the_objective,
    scrape_infolibre,
)
import pandas as pd
import datetime
import time
from fastapi.responses import StreamingResponse
import logging

app = FastAPI()

# Configurar logging
logging.basicConfig(level=logging.INFO)
app.logger = logging.getLogger("ScrapingApp")

# Configuración de rutas estáticas y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

import os
from datetime import datetime

LAST_SCRAPE_FILE = "data/news_scrapped/last_scrape.txt"

def get_last_scrape_date():
    if os.path.exists(LAST_SCRAPE_FILE):
        try:
            with open(LAST_SCRAPE_FILE, "r") as file:
                date_str = file.read().strip()
                if not date_str:
                    return None
                return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None

def set_last_scrape_date():
    os.makedirs(os.path.dirname(LAST_SCRAPE_FILE), exist_ok=True)
    with open(LAST_SCRAPE_FILE, "w") as file:
        file.write(datetime.now().strftime("%Y-%m-%d"))

@app.get("/scraping-logs")
async def scraping_logs():
    def event_stream():
        steps = [
            "Iniciando scraping",
            "Extrayendo noticias de El País",
            "Extrayendo noticias de El Mundo",
            "Extrayendo noticias de El Diario",
            "Extrayendo noticias de La Razón",
            "Extrayendo noticias de The Objective",
            "Extrayendo noticias de InfoLibre",
            "Clasificando noticias",
            "Scraping y clasificación completados"
        ]
        for step in steps:
            yield f"data: {step}\n\n"
            time.sleep(2)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(status_code=200)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Clasificador de Noticias"})


def validate_mode(mode: str):
    """Validate the classification mode."""
    valid_modes = ["basic", "advanced"]
    if mode not in valid_modes:
        raise ValueError(f"Modo inválido. Debe ser uno de {valid_modes}.")


@app.post("/classify/", response_class=HTMLResponse)
async def classify(request: Request, mode: str = Form(...)):
    try:
        validate_mode(mode)  # Ensure valid mode is chosen.
        classified_news = classify_news_combined(use_static_data=True, mode=mode)

        if classified_news.empty:
            return templates.TemplateResponse(
                "classify.html",
                {"request": request, "title": "Resultados", "message": "No se encontraron datos para clasificar."}
            )

        table_data = classified_news.to_dict(orient="records")
        return templates.TemplateResponse(
            "classify.html",
            {"request": request, "title": "Resultados de Clasificación", "table_data": table_data, "mode": mode}
        )
    except ValueError as ve:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "title": "Error", "message": str(ve)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la clasificación: {str(e)}")


@app.post("/scrape-and-classify/", response_class=HTMLResponse)
async def scrape_and_classify(request: Request, mode: str = Form(...)):
    try:
        validate_mode(mode)  # Ensure valid mode is chosen.

        last_scrape_date = get_last_scrape_date()
        today = datetime.now().date()

        if last_scrape_date == today:
            classified_news = classify_news_combined(use_static_data=False, mode=mode)

            if classified_news.empty:
                return templates.TemplateResponse(
                    "scrape_and_classify.html",
                    {"request": request, "title": "Scraping y Clasificación",
                     "message": "No hay datos guardados para clasificar."}
                )

            table_data = classified_news.to_dict(orient="records")
            return templates.TemplateResponse(
                "scrape_and_classify.html",
                {"request": request, "title": "Scraping y Clasificación (Datos Previos)", "table_data": table_data,
                 "mode": mode}
            )

        # Execute scraping logic
        scrapers = {
            "ElPais": scrape_elpais,
            "ElMundo": scrape_el_mundo,
            "ElDiario": scrape_el_diario,
            "LaRazon": scrape_la_razon,
            "TheObjective": scrape_the_objective,
            "InfoLibre": scrape_infolibre,
        }

        combined_data = []
        for source_name, scraper in scrapers.items():
            df = scraper()
            if not df.empty:
                combined_data.append(df)

        if not combined_data:
            return templates.TemplateResponse(
                "scrape_and_classify.html",
                {"request": request, "title": "Scraping y Clasificación",
                 "message": "No se encontraron noticias recientes para procesar."}
            )

        set_last_scrape_date()
        classified_news = classify_news_combined(use_static_data=False, mode=mode)

        if classified_news.empty:
            return templates.TemplateResponse(
                "scrape_and_classify.html",
                {"request": request, "title": "Scraping y Clasificación",
                 "message": "Scraping completado, pero no se encontraron datos para clasificar."}
            )

        table_data = classified_news.to_dict(orient="records")
        return templates.TemplateResponse(
            "scrape_and_classify.html",
            {"request": request, "title": "Scraping y Clasificación", "table_data": table_data, "mode": mode}
        )
    except ValueError as ve:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "title": "Error", "message": str(ve)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el scraping y clasificación: {str(e)}")
