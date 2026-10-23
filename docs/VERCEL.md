# Déploiement de test sur Vercel avec Neon

Vercel sert uniquement à la phase de test. Le code reste portable : sans `DATABASE_URL`, il revient à
SQLite, et le `Dockerfile` permet de passer ensuite chez n'importe quel hébergeur.

## Ce qui change sur Vercel

| Sujet | Serveur classique | Vercel |
|---|---|---|
| Base applicative | SQLite sur un volume | PostgreSQL Neon, via `DATABASE_URL` |
| Messages WhatsApp | Réponse à Meta immédiate, traitement en arrière-plan | Traitement pendant la requête, puis réponse à Meta |
| Point d'entrée | `uvicorn rag_fisc_v2.main:app` | `app.py` à la racine, détecté automatiquement |
| Durée maximale d'une requête | Illimitée | 60 secondes, réglée dans `vercel.json` |
| Région | Celle du serveur | Francfort (`fra1`), à côté de la base Neon `eu-central-1` et de Mistral |

Le mode de traitement est choisi automatiquement : Vercel définit la variable `VERCEL`, qui active le
traitement pendant la requête. Si Meta renvoie le message pendant ce temps, le renvoi est écarté comme doublon.

## Étapes

### 1. Préparer le dépôt

L'ingestion tourne sur votre PC, pas sur Vercel :

```bash
uv run rag-fisc-v2 ingest
```

Versionnez ensuite les fichiers dont le service a besoin : `data/manifest.yaml`, `data/baremes/*.json`
et `data/index_state.json`. Poussez le dépôt sur GitHub. Les PDF et les caches restent exclus par `.gitignore`.

### 2. Créer le projet Vercel

1. Sur vercel.com, importer le dépôt GitHub. Le préréglage FastAPI est détecté, et Python 3.13 est lu dans `.python-version`.
2. Ne pas déployer tout de suite : configurer d'abord la base et les variables.

### 3. Créer la base Neon

1. Dans le projet Vercel, onglet **Storage**, ajouter **Neon** depuis la Marketplace et créer une base.
2. Connecter la base au projet. L'intégration ajoute les variables de connexion au projet.
3. Vérifier que `DATABASE_URL` figure dans **Settings > Environment Variables**. Sinon, copier l'URL de
   connexion **poolée** depuis la console Neon, celle dont l'hôte contient `-pooler`.

Les tables sont créées automatiquement au premier appel.

### 4. Variables d'environnement

Dans **Settings > Environment Variables**, pour l'environnement Production :

| Variable | Valeur |
|---|---|
| `MISTRAL_API_KEY` | clé Mistral |
| `PINECONE_API_KEY` | clé Pinecone |
| `DATABASE_URL` | ajoutée par Neon |
| `USER_HASH_SALT` | longue chaîne aléatoire, à ne plus changer |
| `WHATSAPP_ACCESS_TOKEN` | jeton Meta |
| `WHATSAPP_PHONE_NUMBER_ID` | identifiant du numéro |
| `WHATSAPP_APP_SECRET` | clé secrète de l'application Meta |
| `WHATSAPP_VERIFY_TOKEN` | chaîne de votre choix |
| `ADMIN_TOKEN` | jeton pour `/admin/stats` |

### 5. Déployer et vérifier

Déployer, puis ouvrir `https://<projet>.vercel.app/health`. Le résultat attendu :

```json
{ "statut": "ok", "mistral_configure": true, "pinecone_configure": true,
  "whatsapp_configure": true, "stockage": "postgres", "documents_indexes": 9 }
```

Si `stockage` vaut `sqlite`, `DATABASE_URL` n'est pas définie pour cet environnement. Si `stockage`
commence par `erreur`, l'URL de connexion est incorrecte.

### 6. Brancher WhatsApp

Suivre [WHATSAPP.md](WHATSAPP.md), avec l'URL de rappel `https://<projet>.vercel.app/whatsapp/webhook`.

Utilisez l'URL de **production**. Les déploiements de prévisualisation sont protégés par
l'authentification Vercel par défaut, et Meta ne pourrait pas les appeler.

## Limites à connaître

- **Démarrage à froid.** Après une période sans message, la première réponse prend quelques secondes de plus, le temps de charger l'application.
- **Offre Hobby.** Elle convient à des tests. Pour un service ouvert au public, vérifiez les conditions d'usage de Vercel.
- **Logs.** Les journaux sont dans l'onglet **Logs** du projet. Les numéros de téléphone n'y apparaissent pas.

## Quitter Vercel

1. Garder la même `DATABASE_URL` Neon, ou exporter les tables vers une autre base PostgreSQL.
2. Déployer le `Dockerfile` chez le nouvel hébergeur avec les mêmes variables.
3. Remplacer l'URL du webhook chez Meta.

Aucune modification du code n'est nécessaire.
