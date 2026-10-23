# RAG fiscal Cameroun

Questions-réponses sourcées sur la fiscalité camerounaise : Code Général des Impôts, lois de finances,
circulaires et arrêtés de la DGI, barèmes mensuels. Mistral (OCR, embeddings, Small, Medium),
LlamaIndex et Pinecone.

L'architecture est décrite dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Installation

```bash
uv sync
cp .env.example .env    # puis renseigner MISTRAL_API_KEY et PINECONE_API_KEY
```

Les PDF vont dans `data/raw/` et sont décrits dans `data/manifest.yaml`.

## Ingestion

```bash
uv run rag-fisc-v2 docs                                 # état du corpus
uv run rag-fisc-v2 ingest --doc fiche-is --no-index     # OCR et découpage d'un document, sans Pinecone
uv run rag-fisc-v2 chunks fiche-is                      # relire les morceaux produits
uv run rag-fisc-v2 ingest                               # tout le corpus : OCR, découpage, indexation
```

L'ingestion est reprise là où elle s'est arrêtée, et un document inchangé n'est ni repassé à l'OCR ni réindexé.
Le texte OCR de chaque document est lisible dans `data/ocr/<doc_id>.md`.

## Questions

```bash
uv run rag-fisc-v2 ask "Quel est le taux de l'impôt sur les sociétés ?" --profil expert
uv run rag-fisc-v2 bareme irpp 350000
uv run rag-fisc-v2 analyze "Qu'est-ce qui a changé pour la TVA en 2026 ?"
uv run rag-fisc-v2 eval
```

## API

```bash
uv run rag-fisc-v2 serve
```

Documentation interactive : http://127.0.0.1:8000/docs

## WhatsApp

Le webhook est servi par la même application, sur `/whatsapp/webhook`.
La création du compte Meta, le branchement et l'hébergement sont décrits dans
[docs/WHATSAPP.md](docs/WHATSAPP.md).

Pour un essai depuis votre PC, exposez l'API avec un tunnel HTTPS :

```bash
uv run rag-fisc-v2 serve
ngrok http 8000
```

## Déploiement

Phase de test sur Vercel avec une base Neon : voir [docs/VERCEL.md](docs/VERCEL.md).

Ensuite, avec Docker chez n'importe quel hébergeur :

```bash
docker build -t rag-fisc .
docker run -p 8000:8000 --env-file .env -v rag-fisc-data:/var/lib/rag-fisc rag-fisc
```

Le volume conserve la base SQLite : conversations et suivi d'usage. Avec `DATABASE_URL`, la base
PostgreSQL est utilisée à la place.

## Suivi d'usage

```bash
uv run rag-fisc-v2 stats --days 30
```

## Tests

```bash
uv run pytest
```

Les tests n'appellent ni Mistral, ni Pinecone, ni Meta. Les tests PostgreSQL ne tournent que si
`TEST_DATABASE_URL` désigne une base de test jetable, jamais la base Neon de production.
