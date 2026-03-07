from langchain_community.document_loaders import PyPDFLoader


def document_loader(file_path):
    """
    Load a PDF document and return its pages as documents.

    Args:
        file_path (str): Path to the PDF file to be loaded.

    Returns:
        list: A list of document objects representing the pages of the PDF.
              Returns an empty list if the file cannot be loaded.
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print("Total pages loaded:", len(docs))
        return docs
    except Exception as e:
        print("Failed to load pdf:", e)
        return []


# file_path = "3D_MedDiffusion_A_3D_Medical_Latent_Diffusion_Model_for_Controllable_and_High-Quality_Medical_Image_Generation.pdf"
# document_loader(file_path)
