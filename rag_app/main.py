from data_loader import document_loader
from embeddings import text_splitter, create_documents, save_embeddings
from vector_store import create_qdrant_collection
from retrival import create_index
from llm import llm_response
from langchain_huggingface import HuggingFaceEmbeddings


file_path = "3D_MedDiffusion_A_3D_Medical_Latent_Diffusion_Model_for_Controllable_and_High-Quality_Medical_Image_Generation.pdf"

# Load
docs = document_loader(file_path)

# Chunk
texts = text_splitter(docs)

# Create docs
documents = create_documents(texts)

# Create embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = create_qdrant_collection(embeddings)

# 5 Store vectors
save_embeddings(documents, client, "medical_paper", embeddings)

# 6 Retrieval
index = create_index(client, "medical_paper")

query_engine = index.as_query_engine(similarity_top_k=3)

query = "What is BiFlowNet?"

response = query_engine.query(query)

# Extract context
context = "\n\n".join([node.node.text for node in response.source_nodes])

# 7 LLM
answer = llm_response(context, query)

print("\nRetrieved Context:\n", context)
print("\nFinal Answer:\n", answer)
