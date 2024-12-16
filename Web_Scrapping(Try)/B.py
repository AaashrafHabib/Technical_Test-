import requests
from bs4 import BeautifulSoup

def extract_patent_drawings(patent_url: str, api_key: str):
    """
    Extrait toutes les informations HTML de la page du brevet, y compris les URLs des dessins.
    
    Args:
    patent_url (str): L'URL de la page des dessins du brevet.
    api_key (str): La clé API de ScraperAPI.

    Returns:
    Tuple[str, List[str]]: Le HTML complet de la page et la liste des URLs des dessins du brevet.
    """
    # Utilisation de ScraperAPI pour contourner Cloudflare
    scraperapi_url = f"http://api.scraperapi.com?api_key={api_key}&url={patent_url}"
    
    # Envoi de la requête avec l'API ScraperAPI pour contourner Cloudflare
    response = requests.get(scraperapi_url)
    
    if response.status_code == 200:
        # Récupérer le contenu HTML complet de la page
        html_content = response.content

        # Utiliser BeautifulSoup pour analyser le HTML de la réponse
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Trouver toutes les balises <img> qui contiennent les dessins
        img_tags = soup.find_all("img")
        image_urls = []

        # Extraire l'URL de chaque image
        for img in img_tags:
            img_url = img.get("src")
            if img_url:
                # Vérifiez si l'URL est absolue ou relative
                if img_url.startswith("/"):
                    img_url = f"https://worldwide.espacenet.com{img_url}"
                image_urls.append(img_url)

        return html_content, image_urls
    else:
        print(f"Erreur lors de la récupération de la page : {response.status_code}")
        return None, []

# Exemple d'utilisation pour un brevet donné
patent_url = "https://worldwide.espacenet.com/patent/drawing?channel=espacenet_channel-f4a16409-1a25-44cc-80ce-6316af1a5a4a"
api_key = "2235de32f18ac12e82988d23deae46ff"  # Remplacez ceci par votre clé API

# Extraire le HTML complet de la page et les URLs des dessins du brevet
html_content, image_urls = extract_patent_drawings(patent_url, api_key)

# Si le HTML a été récupéré avec succès
if html_content:
    # Sauvegarder le HTML dans un fichier
    with open("patent_page.html", "wb") as f:
        f.write(html_content)
    print("Le contenu HTML a été sauvegardé dans 'patent_page.html'.")

    # Afficher les URLs des images
    print("\nURLs des dessins extraites :")
    for url in image_urls:
        print(url)

    # Optionnel : Sauvegarder les URLs dans un fichier
    with open("patent_images_urls.txt", "w", encoding="utf-8") as f:
        for url in image_urls:
            f.write(url + "\n")

    print("Les URLs des dessins ont été sauvegardées dans 'patent_images_urls.txt'.")
