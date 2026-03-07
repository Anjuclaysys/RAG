from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
import uuid


def text_splitter(docs):
    """
     Split loaded documents into smaller text chunks.
    Args:
        docs (list): List of LangChain Document objects.

    Returns:
        list: List of chunked Document objects with page content and metadata.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )
    texts = text_splitter.split_documents(docs)
    # Remove empty chunks
    texts = [
        chunk for chunk in texts if chunk.page_content and chunk.page_content.strip()
    ]
    print("Total chunks:", len(texts))
    return texts


def create_documents(texts):
    """
    Convert text chunks into structured document dictionaries.
    Args:
        texts (list): List of chunked Document objects.

    Returns:
        list: List of dictionaries with keys:
            - 'text' (str): Chunk text content
            - 'metadata' (dict): Metadata including chunk index
    """
    documents = []

    for i, chunk in enumerate(texts):

        metadata = chunk.metadata.copy()
        metadata["chunk_index"] = i

        documents.append({"text": chunk.page_content, "metadata": metadata})

    print("Documents prepared:", len(documents))
    return documents


def save_embeddings(documents, client, collection_name, embeddings):
    """
    Generate embeddings for documents and store them in Qdrant.

    Each document text is converted into a vector embedding using the
    provided embeddings model. The vectors along with their payload
    (text and metadata) are stored in the specified Qdrant collection.

    Args:
        documents (list): List of dictionaries containing 'text' and 'metadata'.
        client (QdrantClient): Initialized Qdrant client instance.
        collection_name (str): Name of the Qdrant collection where vectors will be stored.
        embeddings: Embedding model used to generate vector representations.

    Returns:
        None
    """

    points = []

    for doc in documents:

        vector = embeddings.embed_query(doc["text"])

        payload = {"text": doc["text"], "metadata": doc["metadata"]}

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)

    print("Embeddings stored:", len(points))
