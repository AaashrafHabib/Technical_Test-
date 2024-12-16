import time
from playwright.sync_api import sync_playwright

def scrape_full_html(patent_number: str) -> str:
    """
    Scrape l'intégralité du contenu HTML d'une page de brevet en contournant Cloudflare.
    Args:
    patent_number (str): Numéro du brevet à scraper.

    Returns:
    str: Contenu HTML complet de la page.
    """
    # URL de la page du brevet sur Espacenet
    url = f"https://worldwide.espacenet.com/patent/search/family/009541606/publication/{patent_number}?q={patent_number}"

    with sync_playwright() as p:
        # Lancer le navigateur en mode visible (pas headless)
        browser = p.chromium.launch(headless=False)  # Modifier à False pour voir le navigateur
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )  # Définir le User-Agent au niveau du contexte
        page = context.new_page()

        # Accéder à la page du brevet
        page.goto(url, wait_until="domcontentloaded")  # Attendre que le DOM soit complètement chargé

        # Attendre quelques secondes pour laisser le temps à la page de se charger complètement
        time.sleep(5)  # Ajustez si nécessaire en fonction du délai de Cloudflare

        # Ajouter des actions comme le défilement pour simuler un comportement humain
        page.evaluate("window.scrollBy(0, window.innerHeight)")  # Scroll down

        # Attendre un peu pour voir si cela aide à contourner la protection
        time.sleep(3)

        # Extraire tout le contenu HTML de la page
        page_content = page.content()

        browser.close()

    return page_content


# Exemple d'utilisation pour le brevet FR2789320A1
patent_number = "FR2789320A1"

# Scraper tout le contenu HTML de la page
html_content = scrape_full_html(patent_number)

# Sauvegarder le contenu HTML dans un fichier (facultatif)
with open(f"{patent_number}_full_html.html", "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"Le contenu HTML complet a été sauvegardé dans '{patent_number}_full_html.html'.")
