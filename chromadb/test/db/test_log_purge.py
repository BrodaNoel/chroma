from chromadb.test.property import invariants
from chromadb.test.conftest import ClientFactories
import pytest


def test_log_purge(client_factories: ClientFactories) -> None:
    if (
        client_factories._system.settings.chroma_api_impl
        != "chromadb.api.rust.RustBindingsAPI"
    ):
        pytest.skip("This test is only valid for Rust bindings")

    client = client_factories.create_client()

    first_collection = client.create_collection(
        "first_collection", metadata={"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}
    )
    second_collection = client.create_collection(
        "second_collection", metadata={"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}
    )
    collections = [first_collection, second_collection]

    # (Does not trigger a purge)
    for i in range(5):
        first_collection.add(ids=str(i), embeddings=[i, i])

    # (Should trigger a purge)
    for i in range(100):
        second_collection.add(ids=str(i), embeddings=[i, i])

    # The purge of the second collection should not be blocked by the first
    invariants.log_size_below_max(client._system, collections, True)


def test_log_purge_with_multiple_collections(client_factories: ClientFactories) -> None:
    if (
        client_factories._system.settings.chroma_api_impl
        != "chromadb.api.rust.RustBindingsAPI"
    ):
        pytest.skip("This test is only valid for Rust bindings")

    client = client_factories.create_client()
    client.reset()

    first_collection = client.create_collection(
        "first_collection", metadata={"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}
    )
    second_collection = client.create_collection(
        "second_collection", metadata={"hnsw:sync_threshold": 10, "hnsw:batch_size": 10}
    )
    collections = [first_collection, second_collection]

    # (Does not trigger a purge)
    for i in range(15):
        first_collection.add(ids=str(i), embeddings=[i, i])

    # (Should trigger a purge)
    for i in range(25):
        second_collection.add(ids=str(i), embeddings=[i, i])

    invariants.log_size_for_collections_match_expected(
        client._system, collections, True
    )
