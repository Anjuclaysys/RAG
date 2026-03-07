from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def create_index(client, collection_name):
    """
    Connect LlamaIndex to an existing Qdrant collection
    and create a VectorStoreIndex.

    Args:
        client: QdrantClient instance
        collection_name (str): Name of the Qdrant collection

    Returns:
        index: LlamaIndex VectorStoreIndex
    """

    # Disable LLM (retrieval only)
    Settings.llm = None

    # Set embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Connect to Qdrant vector store
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )

    print("Vector index created successfully")

    return index
