"""Script para buscar e baixar papers do ArXiv por categoria.

Filtra por afiliação via query (Anthropic, DeepMind, OpenAI, etc.) e por data
(últimos 60 dias). A afiliação é incluída diretamente na query enviada ao ArXiv,
que filtra no servidor — mais eficiente do que baixar tudo e filtrar localmente.
Limpa data/pdfs/ antes de cada execução.
"""

import json
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import arxiv
import requests

# Configurações
PAPERS_PER_CATEGORY = 30  # busca mais para compensar o filtro de afiliação
PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
INDEX_FILE = Path(__file__).parent.parent / "data" / "papers_index.json"
MAX_DAYS = 60  # janela de tempo para filtro de data

SEARCH_QUERIES: dict[str, str] = {
    "LLM agents": "LLM agents",
    "RAG retrieval augmented generation": "RAG retrieval augmented generation",
    "LLM fine-tuning": "LLM fine-tuning",
    "multi-agent systems LLM collaboration": "multi-agent systems LLM collaboration",
    "LLM evaluation harness benchmarks": "LLM evaluation harness benchmarks",
    "AGI reasoning planning self-correction": "AGI reasoning planning self-correction",
    "LLM inference serving optimization": "LLM inference serving optimization",
    "AI safety alignment language models": "AI safety alignment language models",
}

AFFILIATIONS: list[str] = [
    "Anthropic",
    "Google DeepMind",
    "DeepMind",
    "OpenAI",
    "Meta AI",
    "Microsoft Research",
    "Google Research",
]

# Cláusula OR de afiliações injetada em cada query — filtra no servidor do ArXiv
_AFFIL_CLAUSE = " OR ".join(f'"{a}"' for a in AFFILIATIONS)


def sanitize_filename(title: str) -> str:
    """Converte título em nome de arquivo seguro."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    return safe[:80].strip().replace(" ", "_")


def is_recent(paper: arxiv.Result, max_days: int = MAX_DAYS) -> bool:
    """Retorna True se o paper foi publicado nos últimos max_days dias."""
    if paper.published is None:
        return False
    cutoff = datetime.now(UTC) - timedelta(days=max_days)
    return paper.published >= cutoff


def has_target_affiliation(paper: arxiv.Result) -> bool:
    """Retorna True se o abstract menciona alguma das afiliações alvo."""
    haystack = (paper.summary + " " + " ".join(str(a) for a in paper.authors)).lower()
    return any(aff.lower() in haystack for aff in AFFILIATIONS)


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
    """Busca papers no ArXiv combinando a query de tema com o filtro de afiliação.

    A cláusula de afiliação é injetada diretamente na query Lucene do ArXiv,
    que filtra no servidor — muito mais eficiente do que baixar tudo e filtrar
    localmente.
    """
    combined = f"({query}) AND ({_AFFIL_CLAUSE})"
    client = arxiv.Client()
    search = arxiv.Search(
        query=combined,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


def process_category(
    category_name: str, query: str
) -> tuple[list[dict[str, str | list[str]]], int]:
    """Busca, filtra por data, baixa e indexa papers de uma categoria.

    A filtragem por afiliação já foi feita pelo ArXiv na query.
    Aqui aplicamos apenas o filtro de data local.

    Retorna (lista de metadados dos papers aprovados, total buscado pelo ArXiv).
    """
    print(f"\n=== Categoria: {category_name} ===")
    results = fetch_papers(query, PAPERS_PER_CATEGORY)
    recent = [p for p in results if is_recent(p)]

    print(f"  {len(results)} retornados → {len(recent)} recentes (≤{MAX_DAYS}d)")

    papers: list[dict[str, str | list[str]]] = []

    for i, paper in enumerate(recent, 1):
        arxiv_id = paper.get_short_id()
        filename = f"{sanitize_filename(paper.title)}_{arxiv_id}.pdf"
        dest = PDF_DIR / filename

        print(f"  [{i}/{len(recent)}] {paper.title[:65]}...")

        if dest.exists():
            print("    → Já existe, pulando download.")
            downloaded = True
        else:
            downloaded = download_pdf(paper.pdf_url, dest)
            if downloaded:
                print(f"    → Salvo: {filename}")
            time.sleep(1)

        authors_str = " ".join(str(a) for a in paper.authors)
        haystack = (paper.summary + " " + authors_str).lower()
        matched = [aff for aff in AFFILIATIONS if aff.lower() in haystack]

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
                "afiliacao_detectada": matched,
                "arquivo_local": str(dest) if downloaded else "",
            }
        )

    return papers, len(results)


def main() -> None:
    """Ponto de entrada: limpa pdfs/, filtra e baixa papers, salva índice."""
    # Limpa o diretório de PDFs antes de baixar
    if PDF_DIR.exists():
        shutil.rmtree(PDF_DIR)
        print(f"Diretório {PDF_DIR} limpo.")
    PDF_DIR.mkdir(parents=True)

    all_papers: list[dict[str, str | list[str]]] = []
    total_fetched = 0

    for category_name, query in SEARCH_QUERIES.items():
        papers, fetched = process_category(category_name, query)
        all_papers.extend(papers)
        total_fetched += fetched

    INDEX_FILE.write_text(
        json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    downloaded = sum(1 for p in all_papers if p.get("arquivo_local"))
    unique_ids = len({p["arxiv_id"] for p in all_papers})

    print(f"\n{'=' * 60}")
    print(f"Papers buscados (total):          {total_fetched}")
    print(f"Papers com afiliação-alvo:        {len(all_papers)}")
    print(f"Papers únicos (sem duplicatas):   {unique_ids}")
    print(f"PDFs baixados com sucesso:        {downloaded}")
    print(f"Índice salvo em:                  {INDEX_FILE}")
    print(f"PDFs salvos em:                   {PDF_DIR}")


if __name__ == "__main__":
    main()
