# RAG fiscal Cameroun — Architecture

Document de référence du projet. Il fixe le périmètre, les choix techniques et leurs raisons.

## 1. Périmètre

| Élément | Décision |
|---|---|
| Juridiction | Cameroun, textes nationaux. CEMAC prévu plus tard (champ `juridiction` déjà présent). |
| Corpus | CGI, lois de finances (LFI), circulaires d'application, arrêté, fiche pratique, barèmes |
| Volume | 11 PDF, environ 1 650 pages à passer à l'OCR |
| Historique | Depuis l'exercice 2024 pour les LFI et circulaires. CGI disponible pour 2026 seulement. |
| Langue | Français uniquement |
| Utilisateurs | Experts et grand public, via un paramètre `profil` |
| Canaux | WhatsApp au lancement. Web ensuite, selon l'usage mesuré. |
| Interface | API REST (FastAPI), webhook WhatsApp, ligne de commande pour l'administration |
| Mises à jour | Régulières, ingestion idempotente |

### Inventaire du corpus

Le fichier `data/manifest.yaml` décrit chaque document. Les PDF sont dans `data/raw/`.

| doc_id | Type | Exercice | Pages | Format |
|---|---|---|---|---|
| cgi-2026 | CGI | 2026 | 1 034 | Texte natif sur deux colonnes, sommaire scanné |
| lfi-2024 | LFI | 2024 | 112 | Texte natif de mauvaise qualité |
| lfi-2025 | LFI | 2025 | 110 | Scan |
| lfi-2026 | LFI | 2026 | 151 | Scan, 112 Mo |
| circulaire-2024 | Circulaire | 2024 | 49 | Scan |
| circulaire-2025 | Circulaire | 2025 | 68 | Scan |
| circulaire-2026 | Circulaire | 2026 | 107 | Scan |
| arrete-retenue-source-2026 | Arrêté | 2026 | 3 | Texte natif |
| fiche-is | Fiche pratique | non daté | 3 | Texte natif |
| bareme-irpp | Barème | à confirmer | 70 | Texte natif, tableaux |
| bareme-tdl | Barème | à confirmer | 29 | Texte natif, tableaux |

## 2. Stack

| Rôle | Outil | Remarque |
|---|---|---|
| OCR | Mistral OCR | Tous les documents textuels, pour un Markdown uniforme avec tableaux |
| Embeddings | `mistral-embed` | Dimension 1024 |
| Base vectorielle | Pinecone serverless | Un index, un namespace, métrique cosinus |
| Orchestration | LlamaIndex | Nœuds, sérialisation, vector store Pinecone, requêtes |
| Analyse de la question | Mistral Small | Sortie structurée (Pydantic), avec repli sur des règles |
| Génération | Mistral Medium | Réponse sourcée selon le profil |
| Reranking | Mistral Small, par défaut | Choisi après comparaison avec `bge-reranker-v2-m3`, voir la section 5.4 |
| Barèmes | pdfplumber | Extraction exacte des tableaux, sans OCR |
| API et CLI | FastAPI, Typer | Python 3.13, `uv` |
| Canal WhatsApp | API WhatsApp Cloud de Meta, appelée avec httpx | Guide de mise en service : [WHATSAPP.md](WHATSAPP.md) |
| Stockage applicatif | PostgreSQL (Neon) si `DATABASE_URL` est défini, SQLite sinon | Conversations, préférences, quotas, suivi d'usage |
| Déploiement de test | Vercel et Neon | Voir [VERCEL.md](VERCEL.md). Messages WhatsApp traités pendant la requête. |
| Déploiement ensuite | Docker | Un processus, base Neon ou SQLite sur volume |

Les identifiants des modèles sont dans la configuration (`.env`), jamais dans le code.

**Appels aux modèles de langage par le SDK Mistral.** Le connecteur LLM de LlamaIndex refuse les noms de
modèles qu'il ne connaît pas, ce qui empêcherait de figer une version précise comme Small 4 ou Medium 3.5.
L'analyse et la génération appellent donc le SDK Mistral directement. LlamaIndex reste utilisé pour tout
ce qui touche aux nœuds, aux embeddings et à Pinecone.

## 3. Rôle des sources

- **CGI** : la règle en vigueur. Source principale. Chaque édition annuelle est consolidée.
- **Loi de finances** : ce qui a changé et à partir de quand.
- **Circulaire** : l'interprétation de l'administration, avec des exemples chiffrés.
- **Arrêté** : mesure d'application ponctuelle, par exemple la liste des entités tenues à la retenue à la source.
- **Fiche pratique** : explication simple, utile au grand public.
- **Barème** : montant exact pour un salaire donné. Il n'est pas vectorisé, voir la section 6.

## 4. Pipeline d'ingestion

```
data/raw/*.pdf + data/manifest.yaml
        │
        ├── barèmes ──► pdfplumber ──► data/baremes/<doc_id>.json
        │
        ▼
 1. OCR Mistral par lots de 20 pages ──► data/ocr/<doc_id>/part_NNN.json
        │                                 data/ocr/<doc_id>.json et .md (relecture)
        ▼
 2. Nettoyage : sommaire, numéros de page, en-têtes répétés, césures
        │
        ▼
 3. Découpage structurel selon le type ──► data/chunks/<doc_id>.jsonl
        │
        ▼
 4. Embeddings mistral-embed ──► upsert Pinecone (id "<doc_id>#<n>")
```

### 4.1 OCR

- Les PDF sont découpés en lots de 20 pages, envoyés par l'API Files de Mistral. Le plus gros lot fait 17 Mo.
- Chaque lot est mis en cache dès qu'il est traité. Une interruption reprend au lot suivant.
- Un document dont l'empreinte SHA-256 n'a pas changé n'est jamais repassé à l'OCR.
- Les en-têtes et pieds de page sont séparés du texte par Mistral OCR. Les tableaux sont réinsérés en Markdown.
- Le fichier `data/ocr/<doc_id>.md` permet de relire le texte avant indexation.

### 4.2 Découpage

| Type | Unité | Métadonnées propres |
|---|---|---|
| CGI, corps du Code | Article, avec la hiérarchie Livre > Titre > Sous-titre > Chapitre > Section. Les suites d'un article long reprennent ses sous-titres internes (« Article 7 (suite) — A - Frais généraux > 1- Rémunérations »). | `article` |
| CGI, annexes | Textes reproduits : lois, ordonnances, décrets, arrêtés, décisions, circulaires. Plan propre à chaque texte, sous « ANNEXES DU CGI > PARTIE LEGISLATIVE » ou « PARTIE REGLEMENTAIRE ». | aucun `article` : leurs articles ne sont pas ceux du CGI |
| LFI | Article de la loi (« ARTICLE HUITIÈME »). Chaque article du CGI recopié dans la loi ouvre un sous-morceau. | `article` (numéro dans la loi), `articles_cites` (article du CGI modifié en premier) |
| Circulaire, fiche | Titres du plan (« I- », « A- », titres Markdown), puis regroupement des petites sections | `articles_cites` |
| Arrêté | Article | `article`, `articles_cites` |

Règles communes :

- Taille maximale de 3 500 caractères. Un article long est découpé par paragraphes, et chaque suite commence par « Article N (suite) ».
- Un tableau n'est jamais coupé, sauf s'il dépasse la taille maximale. Il est alors coupé par lignes, avec l'en-tête répété.
- La détection des articles tolère les variantes d'OCR : « ARTICLE 92 BIS », « Art. L106 », « Article 5,- ».
- Les renvois d'un texte vers lui-même (« article 3 de la présente loi », « du présent arrêté ») ne sont pas comptés comme des articles du CGI.

### 4.3 Métadonnées

| Champ | Exemple | Usage |
|---|---|---|
| `doc_id` | `cgi-2026` | Filtrage, suppression |
| `type` | `cgi` | Filtrage, ordre d'autorité |
| `exercice` | `2026`, ou `0` si non daté | Filtrage temporel |
| `titre_document` | `Code Général des Impôts, édition 2026` | Citation, embedding |
| `hierarchie` | `LIVRE PREMIER > TITRE I > CHAPITRE II` | Contexte, embedding |
| `article` | `92 bis` | Recherche exacte |
| `articles_cites` | `["21", "L106"]` | Lien vers le CGI |
| `page_debut`, `page_fin` | `42` | Citation vérifiable |
| `juridiction` | `cameroun` | Ajout futur de la CEMAC |

Le texte vectorisé commence par le titre du document, le plan et le numéro d'article. Un passage
isolé reste ainsi situé dans son contexte.

### 4.4 Mises à jour

- Un document modifié est réindexé : suppression de ses vecteurs par préfixe `<doc_id>#`, puis réinsertion.
- Un document dont les morceaux n'ont pas changé est ignoré, grâce à l'empreinte stockée dans `data/index_state.json`.
- Une nouvelle édition s'ajoute au manifeste avec un nouveau `doc_id`. Les éditions précédentes restent indexées.

## 5. Pipeline de requête

```
Question + profil (+ exercice)
        │
        ▼
 1. Analyse : règles (années, articles, montant) + Mistral Small (reformulation, intention)
        │
        ├── intention = calcul_bareme ──► recherche exacte dans le barème [B]
        ├── intention = hors_sujet ─────► réponse de refus, sans appel au modèle
        ▼
 2. Recherche, pour chaque exercice visé :
    a. articles du CGI cités dans la question (recherche exacte)
    b. passages de LFI, circulaire ou arrêté qui citent ces articles
    c. recherche sémantique filtrée par exercice (30 passages)
    d. commentaires liés aux articles du CGI les mieux classés
    e. famille des articles trouvés : 17 -> 17 bis, 17 ter, 17 quater...
    Les requêtes indépendantes partent en parallèle.
        │
        ▼
 3. Reranking par Mistral Small, seuil minimal, articles demandés toujours conservés et placés en tête
        │
        ▼
 4. Génération (Mistral Medium) : sources numérotées [S1], [S2], barème [B]
        │
        ▼
 Réponse + sources citées + calcul + avertissements
```

### 5.1 Règles temporelles

- Sans précision, l'exercice est le plus récent du corpus.
- Une année citée dans la question, ou le paramètre `exercice`, fixe l'exercice.
- Le modèle ne peut pas ajouter une année ou un article absents de la question.
- Pour une comparaison, chaque exercice est interrogé séparément. « Qu'est-ce qui a changé en 2026 » ajoute l'exercice précédent.
- **CGI manquant pour un exercice.** Pour 2024 et 2025, le CGI 2026 est utilisé avec un avertissement explicite. La LFI et la circulaire de l'exercice font foi.
- Une année absente du corpus produit un avertissement listant les exercices couverts.

### 5.2 Règles de génération

- Chaque affirmation fiscale est suivie de sa source, par exemple [S2].
- Sans source pertinente, la réponse le dit et oriente vers le centre des impôts. Le modèle de génération n'est pas appelé.
- Le CGI fait foi. La circulaire précise, la LFI date les changements.
- Un résultat de barème est repris tel quel, sans recalcul.

### 5.3 Profils

| Profil | Style |
|---|---|
| `expert` | Technique, articles et alinéas précis, conditions, exceptions, entrée en vigueur |
| `grand_public` | Réponse en deux phrases d'abord, sigles expliqués, exemple chiffré, 250 mots maximum, avertissement |

### 5.4 Reranking

**Mistral Small est le reranker par défaut.** Il note chaque passage de 0 à 10. Le choix vient d'une
comparaison sur le corpus réel, pour la question « Quel est le taux de l'impôt sur les sociétés ? » :

| Reranker | Position de l'article 17 du CGI | Durée |
|---|---|---|
| `bge-reranker-v2-m3` (Pinecone) | 20e sur 26 | 1 à 2 s |
| Mistral Small | 2e sur 26 | 2,6 s |

Le reranker de Pinecone sous-estime les articles courts dont le sujet n'apparaît que dans le plan.
Mistral Small coûte environ 0,1 à 0,2 centime par question et n'a pas de quota mensuel.

`RERANK_PROVIDER` permet de changer de reranker : `mistral`, `pinecone`, `auto` (Pinecone jusqu'au
budget mensuel, puis Mistral) ou `none`. Si le reranker échoue, l'ordre de la recherche vectorielle
est conservé. Le reranker utilisé est enregistré pour chaque question.

**Famille d'articles.** L'article 17 bis du CGI, qui fixe le taux réduit de 25 %, n'apparaissait pas dans
les 30 premiers résultats de la recherche sémantique pour cette même question. Les articles bis, ter,
quater complètent ou dérogent à l'article de base : ils sont désormais récupérés ensemble.

### 5.5 Mémoire de conversation

Les échanges sont mémorisés par conversation : numéro WhatsApp, ou `conversation_id` fourni par un client web.

- Les 4 derniers échanges des 2 dernières heures sont transmis à l'analyse de la question.
- Mistral Small reformule une relance (« et pour 2025 ? ») en question autonome. C'est cette question qui sert à la recherche.
- Le modèle ne peut reprendre une année, un article ou un montant que s'il figure dans la conversation.
- La commande « nouveau » efface la conversation.

## 6. Barèmes : un outil de calcul, pas un index

Les barèmes IRPP et TDL contiennent plus de 12 000 lignes chacun, au pas de 250 FCFA. Une ligne
chiffrée est introuvable par recherche sémantique. Les tableaux sont donc extraits en JSON par position
des colonnes. Une question comme « combien d'IRPP sur 350 000 FCFA » est résolue par une recherche exacte.

- IRPP : base de 52 000 à 3 120 750 FCFA par mois. Colonnes IRPP, CAC, CCF, FNE salarial et patronal, CRTV.
- TDL : base de 52 000 à 3 186 750 FCFA par mois.
- Hors de ces bornes, la réponse le signale et s'appuie sur les articles du CGI.

## 7. API

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/ask` | Poser une question |
| `GET` | `/bareme/{irpp\|tdl}?montant=` | Calcul exact par barème |
| `POST` | `/feedback` | Noter une réponse (utile ou non) |
| `GET` | `/documents` | Documents du corpus et état d'indexation |
| `GET` | `/health` | Configuration et état |
| `GET` | `/admin/stats` | Statistiques d'usage, avec l'en-tête `X-Admin-Token` |
| `GET`, `POST` | `/whatsapp/webhook` | Vérification et réception des messages WhatsApp |

## 7 bis. Canal WhatsApp

```
Meta ──POST signé──► /whatsapp/webhook
                        │ vérifie la signature, ignore les doublons
                        │ répond 200 immédiatement
                        ▼
                  tâche d'arrière-plan
                        │ accusé de lecture et indicateur de saisie
                        │ commandes : aide, mode expert, mode simple, nouveau
                        │ quota quotidien par numéro
                        ▼
                  RagService.ask(canal = whatsapp, conversation = numéro)
                        │
                        ▼
                  mise en forme WhatsApp ──► messages de 3 500 caractères au plus
                                         ──► boutons « Oui » / « Non » pour l'avis
```

Différences avec la réponse de l'API :

- consignes de génération plus courtes, sans tableau ni titre ;
- références [S3] renumérotées [1], [2] dans l'ordre d'apparition, avec une liste de sources courte en fin de message ;
- références inventées par le modèle supprimées ;
- mention « information indicative » courte pour le grand public, et message d'accueil complet au premier contact.

## 7 ter. Suivi d'usage

Chaque question est enregistrée en base, avec l'utilisateur anonymisé, le canal, le profil, l'intention,
les exercices, la latence, les jetons Mistral, le reranker utilisé, l'erreur éventuelle et l'avis.
La commande `rag-fisc-v2 stats` et la route `/admin/stats` agrègent ces données. Elles serviront à
décider du passage au web et à enrichir le jeu d'évaluation avec de vraies questions.

Requête `/ask` :

```json
{ "question": "Quel est le taux de l'IS ?", "profil": "grand_public", "exercice": null, "debug": false }
```

Réponse :

```json
{
  "reponse": "… [S1] …",
  "profil": "grand_public",
  "exercices": [2026],
  "sources": [
    { "ref": "S1", "doc_id": "cgi-2026", "type": "cgi", "article": "17", "page_debut": 21, "extrait": "…", "citee": true }
  ],
  "calcul": null,
  "avertissements": ["…"]
}
```

## 8. Évaluation

Le fichier `eval/questions.yaml` contient les questions annotées. La commande `rag-fisc-v2 eval` vérifie :

- l'intention détectée ;
- la présence de l'article attendu dans les passages retrouvés ;
- la présence du document attendu ;
- avec `--generate`, la présence de valeurs attendues dans la réponse.

Le jeu doit atteindre une trentaine de questions validées par un fiscaliste.

## 9. Structure du code

```
src/rag_fisc_v2/
  config.py            paramètres (pydantic-settings, .env)
  domain.py            types de documents, profils, Chunk
  clients.py           clients Mistral et Pinecone, créés à la demande
  storage.py           SQLite : conversations, préférences, quotas, suivi d'usage
  channels/whatsapp/   webhook, client Graph API, traitement des messages, mise en forme
  cli.py               ligne de commande
  main.py              application FastAPI
  eval.py              évaluation
  api/routes.py        routes HTTP
  schemas/api.py       modèles d'entrée et de sortie
  services/rag.py      orchestration question -> réponse
  ingestion/           manifeste, OCR, nettoyage, renvois, découpage, barèmes, indexation, pipeline
  retrieval/           analyse de la question, recherche, reranking
  generation/          consignes et génération
data/
  manifest.yaml        inventaire versionné
  raw/                 PDF sources (hors git)
  ocr/ chunks/ baremes/ index_state.json   fichiers générés (hors git)
eval/questions.yaml    jeu d'évaluation
tests/                 tests sans appel réseau
```

## 10. Avancement

| Étape | État |
|---|---|
| Configuration, manifeste, structure | Fait |
| Barèmes extraits et testés | Fait : 12 276 lignes IRPP, 12 540 lignes TDL |
| OCR du corpus | Fait : 1 637 pages, OCR 4.1, environ 6,50 $ |
| Découpage | Fait et vérifié sur le vrai CGI : 961 articles, annexes séparées |
| Indexation Pinecone | Fait : 3 714 vecteurs, index `rag-fisc-cm` |
| Recherche, reranking, génération | Fait et testé de bout en bout |
| Évaluation | 13 questions, 20 contrôles sur 20 réussis au dernier passage. Jeu à valider par un fiscaliste et à porter à 30 questions. |
| Base Neon, Pinecone, WhatsApp | Configurés et vérifiés en local |
| Déploiement Vercel, webhook Meta | À faire |

## 11. Points ouverts

- Déployer sur Vercel et brancher le webhook chez Meta, voir [VERCEL.md](VERCEL.md) et [WHATSAPP.md](WHATSAPP.md).
- Régénérer avant l'ouverture au public les secrets communiqués pendant la mise en place : mot de passe Neon, clés Pinecone et Mistral, jeton WhatsApp, clé secrète Meta.
- CGI 2024 et 2025 absents : à ajouter si disponibles pour un historique complet.
- Exercice d'application des barèmes IRPP et TDL à confirmer.
- Date de la fiche IS à confirmer : sur le taux de l'IS, elle est cohérente avec les articles 17 et 17 bis du CGI 2026.
- Latence : environ 20 s depuis un PC au Cameroun, dont 12 s de recherche. À mesurer depuis Vercel (Francfort).
