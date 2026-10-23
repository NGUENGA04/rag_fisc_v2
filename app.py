"""Point d'entrée pour Vercel, qui cherche une variable `app` dans app.py à la racine.

Le dossier src/ est placé en tête du chemin d'import : le code du dépôt est
utilisé tel quel, avec ses fichiers de données dans data/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_fisc_v2.main import app  # noqa: E402

__all__ = ["app"]
