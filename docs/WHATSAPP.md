# Mise en service sur WhatsApp

Le service utilise l'API WhatsApp Cloud de Meta, sans intermédiaire. Ce guide va du compte Meta
au premier message reçu.

## 1. Ce qu'il faut avant de commencer

- Un compte Facebook personnel, pour accéder à Meta for Developers.
- Un numéro de téléphone dédié au service, qui ne soit pas déjà utilisé sur l'application WhatsApp.
  Pour les tests, Meta fournit un numéro de test gratuit.
- Le service déployé sur une adresse HTTPS publique (voir la section 5). Pour un essai depuis
  votre PC, un tunnel comme ngrok suffit.

## 2. Créer l'application Meta

1. Sur developers.facebook.com, créer une application de type « Business ».
2. Ajouter le produit **WhatsApp**. Meta crée ou associe un compte WhatsApp Business et propose un numéro de test.
3. Dans **WhatsApp > Configuration de l'API**, relever :
   - le **Phone number ID** : à mettre dans `WHATSAPP_PHONE_NUMBER_ID` ;
   - le **jeton d'accès temporaire** : valable 24 heures, pour les premiers essais, dans `WHATSAPP_ACCESS_TOKEN`.
4. Ajouter jusqu'à cinq numéros destinataires autorisés pour les tests, dont le vôtre.
5. Dans **Paramètres de l'application > Général**, relever la **clé secrète** : à mettre dans `WHATSAPP_APP_SECRET`.
6. Choisir une chaîne aléatoire pour `WHATSAPP_VERIFY_TOKEN`.

## 3. Brancher le webhook

1. Déployer le service avec ces variables, puis vérifier que `GET /health` renvoie `"whatsapp_configure": true`.
2. Dans **WhatsApp > Configuration**, renseigner :
   - URL de rappel : `https://<votre-domaine>/whatsapp/webhook`
   - Jeton de vérification : la valeur de `WHATSAPP_VERIFY_TOKEN`
3. Valider. Meta appelle l'URL et le service répond au défi.
4. S'abonner au champ **messages**.
5. Envoyer « Bonjour » au numéro depuis un téléphone autorisé. Le message d'accueil doit arriver.

## 4. Passer en production

1. **Jeton permanent.** Le jeton temporaire expire en 24 heures. Dans le Business Manager, créer un
   utilisateur système, lui attribuer l'application et générer un jeton avec les permissions
   `whatsapp_business_messaging` et `whatsapp_business_management`.
2. **Vérification de l'entreprise** dans le Business Manager, exigée pour dépasser les limites d'envoi de départ.
3. **Numéro réel et nom d'affichage**, validés par Meta.
4. **Coûts.** Les réponses envoyées dans les 24 heures qui suivent un message de l'utilisateur relèvent du
   service client. D'après la grille de Meta, elles ne sont pas facturées, mais vérifiez la grille en vigueur
   pour le Cameroun. Le service n'envoie jamais de message de sa propre initiative.

## 5. Hébergement

**Phase de test : Vercel avec une base Neon.** La procédure est dans [VERCEL.md](VERCEL.md).
La suite de cette section concerne l'hébergement après la phase de test.

Le service a besoin :

- d'une adresse HTTPS publique et toujours active : Meta renvoie un message non acquitté, et un serveur en veille répondrait trop tard ;
- d'une base persistante pour les conversations et le suivi d'usage : la base Neon de la phase de test
  (`DATABASE_URL`), ou SQLite sur un volume (`/var/lib/rag-fisc`) ;
- d'environ 512 Mo de mémoire. Le calcul lourd est fait chez Mistral et Pinecone.

Le `Dockerfile` à la racine fonctionne chez tous les hébergeurs qui acceptent Docker. Deux options :

| Option | Pour | Contre |
|---|---|---|
| Plateforme comme Railway, Render (offre payante) ou Fly.io | Déploiement depuis GitHub, HTTPS automatique, volumes | Coût mensuel un peu plus élevé |
| Petit serveur privé (VPS) | Le moins cher, contrôle total | HTTPS, mises à jour et sauvegardes à gérer soi-même |

**Recommandation pour le lancement : une plateforme avec volume persistant.** Évitez les offres gratuites qui
mettent le service en veille. Le premier message après une veille arriverait avec plusieurs dizaines de
secondes de retard.

## 6. Commandes reconnues dans la conversation

| Message | Effet |
|---|---|
| bonjour, salut | Message d'accueil |
| aide, menu, ? | Exemples et commandes |
| mode expert | Réponses techniques, mémorisé pour ce numéro |
| mode simple | Réponses courtes et expliquées |
| nouveau | Oublie la conversation en cours |
| supprimer mes données | Efface tout ce qui concerne ce numéro : historique, préférences, questions, compteurs |

Après chaque réponse sourcée, deux boutons « Oui » et « Non » recueillent l'avis de l'utilisateur.

## 7. Garde-fous

- **Signature.** Chaque appel du webhook est vérifié avec `WHATSAPP_APP_SECRET`. Un appel non signé est rejeté.
- **Doublons.** Un message renvoyé par Meta n'est traité qu'une fois.
- **Quota.** Au plus `WHATSAPP_DAILY_LIMIT_PER_USER` questions par numéro et par jour, 30 par défaut.
- **Vie privée.** Les numéros ne sont jamais stockés. La base ne contient qu'un hachage salé par `USER_HASH_SALT`.
- **Messages non textuels.** Notes vocales, images et documents reçoivent une réponse polie demandant un message écrit.

## 8. Suivre l'usage

```bash
uv run rag-fisc-v2 stats --days 30
```

Ou, sur le serveur déployé :

```bash
curl -H "X-Admin-Token: <ADMIN_TOKEN>" https://<votre-domaine>/admin/stats?days=30
```

Les chiffres donnés sont les questions par jour, les utilisateurs distincts, les avis, les jetons Mistral
consommés et les reranks Pinecone du mois. Ce sont les éléments pour décider d'une version web.
