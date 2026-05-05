from app.rag.vector_store import VectorStore, cosine_similarity


def test_cosine_similarity_orders_related_vectors():
    assert cosine_similarity([1, 0], [1, 0]) > cosine_similarity([1, 0], [0, 1])


def test_vector_store_adds_and_searches(tmp_path):
    store = VectorStore(tmp_path / "store.json")
    document = store.add_document(
        name="demo.md",
        chunks=["apple content", "banana content"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        embedding_model="fake-embedding",
    )

    hits = store.search([0.9, 0.1], top_k=1)

    assert document["chunk_count"] == 2
    assert hits[0].text == "apple content"
    assert hits[0].document_name == "demo.md"


def test_vector_store_deletes_document(tmp_path):
    store = VectorStore(tmp_path / "store.json")
    document = store.add_document(
        name="demo.md",
        chunks=["content"],
        embeddings=[[1.0, 0.0]],
        embedding_model="fake-embedding",
    )

    assert store.delete_document(document["id"]) is True
    assert store.stats() == {"documents": 0, "chunks": 0}
    assert store.search([1.0, 0.0], top_k=3) == []

