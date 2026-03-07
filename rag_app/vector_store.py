from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings


def create_qdrant_collection(embeddings, collection_name="medical_paper"):
    """
    Create a Qdrant collection if it does not exist.

    Args:
        embeddings: Embedding model
        collection_name (str): Name of the Qdrant collection

    Returns:
        client: Qdrant client
    """

    # Initialize Qdrant client (in-memory)
    # client = QdrantClient(":memory:")
    client = QdrantClient(path="./qdrant_storage")

    # Determine embedding vector size
    vector_size = len(embeddings.embed_query("test"))

    # Create collection if not exists
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    print(f"Collection '{collection_name}' ready.")

    return client
