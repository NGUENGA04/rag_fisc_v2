import pytest

from rag_fisc_v2.config import get_settings


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Chaque test utilise sa propre base SQLite, jamais data/app.db ni la base Neon du .env."""
    settings = get_settings()
    previous = (settings.app_db_path, settings.database_url)
    settings.app_db_path = tmp_path / "app.db"
    settings.database_url = None
    yield settings.app_db_path
    settings.app_db_path, settings.database_url = previous
