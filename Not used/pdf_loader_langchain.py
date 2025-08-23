# from langchain_community.document_loaders import UnstructuredPDFLoader

# def extract_text_with_unstructured(pdf_path):
#     """Extracts text content from a PDF file using UnstructuredPDFLoader."""
#     try:
#         loader = UnstructuredPDFLoader(pdf_path)
#         documents = loader.load()
#         all_text = ""
#         for doc in documents:
#             all_text += doc.page_content + "\n\n"
#         return all_text
#     except Exception as e:
#         print(f"Error reading PDF with UnstructuredPDFLoader: {e}")
#         return ""

# if __name__ == "__main__":
#     pdf_file_path = "./data/finance.pdf"
#     raw_text = extract_text_with_unstructured(pdf_file_path)
#     if raw_text:
#         print("Extracted Text (using UnstructuredPDFLoader):\n")
#         print(raw_text)  # Print the entire extracted text
#     else:
#         print("No text extracted from the PDF using UnstructuredPDFLoader.")

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_with_unstructured(pdf_path):
    """Extracts text content from a PDF file using UnstructuredPDFLoader."""
    try:
        loader = UnstructuredPDFLoader(pdf_path)
        documents = loader.load()
        all_text = ""
        for doc in documents:
            all_text += doc.page_content + "\n\n"
        return all_text, documents
    except Exception as e:
        print(f"Error reading PDF with UnstructuredPDFLoader: {e}")
        return "", []

if __name__ == "__main__":
    pdf_file_path = "./data/academics.pdf"
    raw_text, documents = extract_text_with_unstructured(pdf_file_path)

    if raw_text:
        print("Extracted Text (using UnstructuredPDFLoader):\n")
        # print(raw_text)  # Print the entire extracted text (commented out now)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # You can adjust this value
            chunk_overlap=200,  # You can adjust this value
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(raw_text)

        print("\n--- Chunks (using RecursiveCharacterTextSplitter) ---")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: \"{chunk}\"")
            print("-" * 40)

    else:
        print("No text extracted from the PDF using UnstructuredPDFLoader.")