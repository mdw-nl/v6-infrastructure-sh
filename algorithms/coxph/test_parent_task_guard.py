import jwt
import pandas as pd
import pytest
from coxph import ensure_spawned_by_central
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


class _StubTaskClient:
    """Minimal stand-in for AlgorithmClient.task, exposing only .get()."""

    def __init__(self, task_record: dict | None = None, raise_on_get: bool = False):
        self._task_record = task_record
        self._raise_on_get = raise_on_get

    def get(self, task_id: int) -> dict:
        if self._raise_on_get:
            raise RuntimeError("server unreachable")
        return self._task_record


class _StubClient:
    """Minimal stand-in for a real (non-mock) AlgorithmClient -- just enough
    for ensure_spawned_by_central to decode a token and look up a task,
    without needing a running vantage6 server."""

    def __init__(self, token: str, task_record: dict | None = None, raise_on_get: bool = False):
        self._access_token = token
        self.task = _StubTaskClient(task_record, raise_on_get)


def _token(task_id=42) -> str:
    # Unverified decode is all ensure_spawned_by_central does (matching how
    # AlgorithmClient itself reads its own identity), so a plain unsigned
    # JWT with the expected claim shape is a faithful stand-in.
    return jwt.encode({"sub": {"task_id": task_id}}, key="irrelevant", algorithm="HS256")


def test_mock_algorithm_client_is_exempt() -> None:
    mock_client = MockAlgorithmClient(
        datasets=[[{"database": pd.DataFrame(), "input_data": {}}]], module="coxph"
    )
    ensure_spawned_by_central(mock_client, "perform_iteration")  # must not raise


def test_passes_when_task_has_a_parent() -> None:
    client = _StubClient(_token(), task_record={"id": 42, "parent": 7})
    ensure_spawned_by_central(client, "perform_iteration")  # must not raise


def test_fails_closed_when_task_has_no_parent() -> None:
    client = _StubClient(_token(), task_record={"id": 42, "parent": None})
    with pytest.raises(RuntimeError, match="sub-task"):
        ensure_spawned_by_central(client, "perform_iteration")


def test_fails_closed_when_token_is_undecodable() -> None:
    client = _StubClient("not-a-real-jwt-token", task_record={"id": 42, "parent": 7})
    with pytest.raises(RuntimeError, match="provenance"):
        ensure_spawned_by_central(client, "perform_iteration")


def test_fails_closed_when_task_lookup_fails() -> None:
    client = _StubClient(_token(), raise_on_get=True)
    with pytest.raises(RuntimeError, match="provenance"):
        ensure_spawned_by_central(client, "perform_iteration")


def test_accepts_parent_id_as_an_alternative_key() -> None:
    # Defends against either the real server's field being named "parent" or
    # "parent_id" -- confirmed via the mock client's own source to be
    # "parent", but this should not be brittle to the exact key name.
    client = _StubClient(_token(), task_record={"id": 42, "parent_id": 7})
    ensure_spawned_by_central(client, "perform_iteration")  # must not raise
