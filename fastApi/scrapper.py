import time

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import logging
def scrape_elpais(logger= None):
    base_url = "https://elpais.com/noticias/politica/"
    if logger:
        logger.info("Iniciando scraping para El País...")

    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []

        # Seleccionar los links de las noticias en portada
        links = soup.select("body > main > div > div.b-b.b-au_b > article > header > h2 > a")

        for link in links:
            if len(articles) >= 5:
                break

            href = link.get('href')

            # Revisar si es un enlace válido
            if href and not link.select("span[name='elpais_ico']._pr"):
                full_url = href if href.startswith("http") else f"https://elpais.com{href}"

                # Entrar a la noticia para extraer el contenido
                news_response = requests.get(full_url)
                news_response.raise_for_status()
                news_soup = BeautifulSoup(news_response.text, 'html.parser')

                # Extraer el título
                title_tag = news_soup.select_one("article > header > div > h1")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Extraer los párrafos específicos
                paragraphs = news_soup.select("body > article > div.a_c.clearfix > p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs[:3])

                # Limpiar el contenido
                content = clean_content(content)

                # Agregar la noticia a la lista
                articles.append({
                    "title": title,
                    "content": content,
                    "url": full_url
                })

                time.sleep(3)


        # Convertir los resultados en un DataFrame
        df = pd.DataFrame(articles)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")

        return pd.DataFrame()


def scrape_el_mundo(logger= None):
    base_url = "https://www.elmundo.es/t/po/politica.html"
    if logger:
        logger.info("Iniciando scraping para El Mundo...")

    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []

        # Seleccionar los links de las noticias en portada
        links = soup.select("a.ue-c-cover-content__link")

        for link in links:
            if len(articles) >= 5:
                break

            href = link.get('href')

            # Revisar si es un enlace válido y no Premium
            if href and not link.find_parent().select("svg.ue-c-cover-content__icon-premium"):
                full_url = href if href.startswith("http") else f"https://www.elmundo.es{href}"

                # Entrar a la noticia para extraer el contenido
                news_response = requests.get(full_url)
                news_response.raise_for_status()
                news_soup = BeautifulSoup(news_response.text, 'html.parser')

                # Extraer el título
                title_tag = news_soup.select_one("h1.ue-c-article__headline")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Extraer los párrafos específicos
                paragraphs = news_soup.select("p.ue-c-article__paragraph")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs[:3])

                # Limpiar el contenido
                content = clean_content(content)

                # Agregar la noticia a la lista
                articles.append({
                    "title": title,
                    "content": content,
                    "url": full_url
                })

                time.sleep(3)


        # Convertir los resultados en un DataFrame
        df = pd.DataFrame(articles)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")
        return pd.DataFrame()


def scrape_el_diario(logger= None):
    base_url = "https://www.eldiario.es/politica/"
    if logger:
        logger.info("Iniciando scraping para El Diario...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []

        # Seleccionar los links de las noticias en portada
        links = soup.select("main h2 > a")

        for link in links:
            if len(articles) >= 5:
                break

            href = link.get('href')

            # Revisar si es un enlace válido
            if href:
                full_url = href if href.startswith("http") else f"https://www.eldiario.es{href}"

                # Entrar a la noticia para extraer el contenido
                news_response = requests.get(full_url)
                news_response.raise_for_status()
                news_soup = BeautifulSoup(news_response.text, 'html.parser')

                # Extraer el título
                title_tag = news_soup.select_one("h1.title")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Extraer los párrafos específicos
                paragraphs = news_soup.select("main .article-text")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs[:3])

                # Limpiar el contenido
                content = clean_content(content)

                # Agregar la noticia a la lista
                articles.append({
                    "title": title,
                    "content": content,
                    "url": full_url
                })

                time.sleep(3)


        # Convertir los resultados en un DataFrame
        df = pd.DataFrame(articles)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")
        return pd.DataFrame()


def scrape_la_razon(logger= None):
    base_url = "https://www.larazon.es/tags/politica/"
    if logger:
        logger.info("Iniciando scraping para La Razon...")

    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []

        # Seleccionar los links de las noticias en portada
        links = soup.select("header > h2 > a")

        for link in links:
            if len(articles) >= 5:
                break

            href = link.get('href')

            # Revisar si es un enlace válido
            if href:
                full_url = href if href.startswith("http") else f"https://www.larazon.es{href}"

                # Entrar a la noticia para extraer el contenido
                news_response = requests.get(full_url)
                news_response.raise_for_status()
                news_soup = BeautifulSoup(news_response.text, 'html.parser')

                # Extraer el título
                title_tag = news_soup.select_one("h1.article-main__title")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Extraer los párrafos específicos
                paragraphs = news_soup.select("#intext > p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs[:3])

                # Limpiar el contenido
                content = clean_content(content)

                # Agregar la noticia a la lista
                articles.append({
                    "title": title,
                    "content": content,
                    "url": full_url
                })

                time.sleep(3)


        # Convertir los resultados en un DataFrame
        df = pd.DataFrame(articles)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")
        return pd.DataFrame()

def scrape_infolibre(logger= None):
    base_url = "https://www.infolibre.es/politica/"
    if logger:
        logger.info("Iniciando scraping para InfoLibre...")

    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []

        # Seleccionar los links de las noticias en portada
        links = soup.select("div.header-dst > div > h2 > a")

        for link in links:
            if len(articles) >= 5:
                break

            href = link.get('href')

            # Revisar si es un enlace válido
            if href:
                full_url = href if href.startswith("http") else f"https://www.infolibre.es{href}"

                # Entrar a la noticia para extraer el contenido
                news_response = requests.get(full_url)
                news_response.raise_for_status()
                news_soup = BeautifulSoup(news_response.text, 'html.parser')

                # Comprobar si la noticia es VIP
                vip_check = news_soup.select_one("aside.paywall")
                if vip_check:
                    continue

                # Extraer el título
                title_tag = news_soup.select_one("h1.title")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Extraer los párrafos específicos
                paragraphs = news_soup.select("div.second-col > p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs[:3])

                # Agregar la noticia a la lista
                articles.append({
                    "title": title,
                    "content": content,
                    "url": full_url
                })

                time.sleep(3)


        # Convertir los resultados en un DataFrame
        df = pd.DataFrame(articles)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")
        return pd.DataFrame()



def scrape_the_objective(logger= None):
    base_url = "https://theobjective.com/espana/politica/"
    if logger:
        logger.info("Iniciando scraping para The Objective...")

    try:
        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []

        # Seleccionar los links de las noticias en portada
        links = soup.select("div.tno-general__main__posts h2 > a")

        for link in links:
            if len(articles) >= 5:
                break

            href = link.get('href')

            # Revisar si es un enlace válido
            if href:
                full_url = href if href.startswith("http") else f"https://theobjective.com{href}"

                # Entrar a la noticia para extraer el contenido
                news_response = requests.get(full_url)
                news_response.raise_for_status()
                news_soup = BeautifulSoup(news_response.text, 'html.parser')

                # Extraer el título
                title_tag = news_soup.select_one("h1.tno-general-single__article__header__title")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Extraer los párrafos específicos
                paragraphs = news_soup.select("div.tno-general-single__article__main__content > p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs[:3])

                # Limpiar el contenido
                content = clean_content(content)

                # Agregar la noticia a la lista
                articles.append({
                    "title": title,
                    "content": content,
                    "url": full_url
                })
                time.sleep(3)

        # Convertir los resultados en un DataFrame
        df = pd.DataFrame(articles)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error durante la solicitud: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado: {e}")
        return pd.DataFrame()




def clean_content(content):
    """
    Remove unwanted phrases and promotional content from the text.
    """
    unwanted_phrases = [
        "Aceptar cookies",
        r"\[.*?newsletter.*?\]",  # Remove promotional content in brackets
        r"Si quiere suscribirse, puede hacerlo.*?enlace"
    ]
    for phrase in unwanted_phrases:
        content = re.sub(phrase, "", content, flags=re.IGNORECASE)
    return content.strip()



def add_source(df, source_name):
    df['Fuente'] = source_name
    return df

# Ejecutar los scrapers
if __name__ == "__main__":


    # Obtener los DataFrames
    el_pais_df = scrape_elpais()
    el_mundo_df = scrape_el_mundo()
    el_diario_df = scrape_el_diario()
    la_razon_df = scrape_la_razon()
    the_objective_df = scrape_the_objective()
    infolibre_df = scrape_infolibre()

    # Limpiar y agregar la fuente a los DataFrames
    el_pais_df["content"] = el_pais_df["content"].apply(clean_content)
    el_pais_df = add_source(el_pais_df, "El País")

    el_mundo_df["content"] = el_mundo_df["content"].apply(clean_content)
    el_mundo_df = add_source(el_mundo_df, "El Mundo")

    el_diario_df["content"] = el_diario_df["content"].apply(clean_content)
    el_diario_df = add_source(el_diario_df, "El Diario")

    la_razon_df["content"] = la_razon_df["content"].apply(clean_content)
    la_razon_df = add_source(la_razon_df, "La Razón")

    the_objective_df["content"] = the_objective_df["content"].apply(clean_content)
    the_objective_df = add_source(the_objective_df, "The Objective")

    infolibre_df["content"] = infolibre_df["content"].apply(clean_content)
    infolibre_df = add_source(infolibre_df, "InfoLibre")

    # Guardar los DataFrames en archivos CSV
    el_pais_df.to_csv("elpais_news.csv", index=False)
    el_mundo_df.to_csv("elmundo_news.csv", index=False)
    el_diario_df.to_csv("eldiario_news.csv", index=False)
    la_razon_df.to_csv("larazon_news.csv", index=False)
    the_objective_df.to_csv("theobjective_news.csv", index=False)
    infolibre_df.to_csv("infolibre_news.csv", index=False)


