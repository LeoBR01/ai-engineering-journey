"""Script para buscar e baixar papers do ArXiv por categoria."""

import json
import time
from pathlib import Path

import arxiv
import requests

# Configurações
PAPERS_PER_CATEGORY = 10
PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
INDEX_FILE = Path(__file__).parent.parent / "data" / "papers_index.json"

SEARCH_QUERIES: dict[str, str] = {
    "LLM agents": "LLM agents",
    "RAG retrieval augmented generation": "RAG retrieval augmented generation",
    "LLM fine-tuning": "LLM fine-tuning",
}


def sanitize_filename(title: str) -> str:
    """Converte título em nome de arquivo seguro."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    return safe[:80].strip().replace(" ", "_")


def download_pdf(url: str, dest: Path) -> bool:
    """Baixa um PDF dado uma URL e salva em dest. Retorna True se bem-sucedido."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return True
    except requests.RequestException as e:
        print(f"  [ERRO] Falha ao baixar {url}: {e}")
        return False


def fetch_papers(query: str, max_results: int) -> list[arxiv.Result]:
    """Busca papers no ArXiv por query, ordenados por data de submissão."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


def process_category(
    category_name: str, query: str
) -> list[dict[str, str | list[str]]]:
    """Busca, baixa e indexa papers de uma categoria. Retorna lista de metadados."""
    print(f"\n=== Categoria: {category_name} ===")
    results = fetch_papers(query, PAPERS_PER_CATEGORY)
    papers: list[dict[str, str | list[str]]] = []

    for i, paper in enumerate(results, 1):
        arxiv_id = paper.get_short_id()
        filename = f"{sanitize_filename(paper.title)}_{arxiv_id}.pdf"
        dest = PDF_DIR / filename

        print(f"  [{i}/{len(results)}] {paper.title[:70]}...")

        if dest.exists():
            print("    -> Já existe, pulando download.")
            downloaded = True
        else:
            pdf_url = paper.pdf_url
            print(f"    -> Baixando: {pdf_url}")
            downloaded = download_pdf(pdf_url, dest)
            if downloaded:
                print(f"    -> Salvo em: {filename}")
            time.sleep(1)  # respeita rate limit do ArXiv

        papers.append(
            {
                "categoria": category_name,
                "titulo": paper.title,
                "autores": [str(a) for a in paper.authors],
                "resumo": paper.summary.replace("\n", " "),
                "data_publicacao": (
                    paper.published.isoformat() if paper.published else ""
                ),
                "arxiv_id": arxiv_id,
                "arquivo_local": str(dest) if downloaded else "",
            }
        )

    return papers


def main() -> None:
    """Ponto de entrada: busca papers de todas as categorias e salva o índice."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    all_papers: list[dict[str, str | list[str]]] = []

    for category_name, query in SEARCH_QUERIES.items():
        papers = process_category(category_name, query)
        all_papers.extend(papers)

    INDEX_FILE.write_text(
        json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    downloaded = sum(1 for p in all_papers if p.get("arquivo_local"))
    print(f"\n{'=' * 60}")
    print(f"Total de papers processados: {len(all_papers)}")
    print(f"PDFs baixados com sucesso:   {downloaded}")
    print(f"Índice salvo em:             {INDEX_FILE}")
    print(f"PDFs salvos em:              {PDF_DIR}")


if __name__ == "__main__":
    main()
