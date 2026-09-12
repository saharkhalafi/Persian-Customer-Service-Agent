import ast
from inspect import getsource
from pathlib import Path

from app.core.config import ENVIRONMENT
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.feedback_repository import FeedbackRepository


ROOT = Path(__file__).resolve().parents[1]


def _literal(node: ast.AST | None) -> object:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    return ast.unparse(node)


def _revision_ids() -> list[tuple[object, object]]:
    found: list[tuple[object, object]] = []
    for path in (ROOT / "alembic" / "versions").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        revision = None
        down_revision = None
        for node in tree.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            if node.target.id == "revision":
                revision = _literal(node.value)
            elif node.target.id == "down_revision":
                down_revision = _literal(node.value)
        if revision is not None:
            found.append((revision, down_revision))
    return found


def test_alembic_has_single_linear_head():
    assert _revision_ids() == [("0001_initial", None)]


def test_repositories_do_not_create_tables():
    assert "CREATE TABLE" not in getsource(ConversationRepository)
    assert "CREATE TABLE" not in getsource(FeedbackRepository)
    database_source = (ROOT / "app" / "core" / "database.py").read_text(encoding="utf-8")
    assert "CREATE INDEX" not in database_source
    assert "CREATE TABLE" not in database_source


def test_environment_defaults_to_a_known_value():
    assert ENVIRONMENT in {"development", "test", "production"}
