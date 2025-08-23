# # import os

# # def extract_text_from_pdf(pdf_path):
# #     """
# #     Extracts text from a PDF file using crude built-in decoding methods.
# #     Only works for text-based PDFs.
# #     """
# #     print(f"[INFO] Attempting to extract text from: {pdf_path}")
# #     text = ""
# #     with open(pdf_path, 'rb') as file:
# #         content = file.read()
# #         try:
# #             text = content.decode('latin1')
# #             print("[DEBUG] Decoded PDF content using 'latin1'.")
# #         except UnicodeDecodeError:
# #             text = content.decode('utf-8', errors='ignore')
# #             print("[DEBUG] Decoded PDF content using 'utf-8' with ignore errors.")
    
# #     print(f"[INFO] Total characters extracted: {len(text)}")
# #     return text

# # def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
# #     """
# #     Splits the extracted text into overlapping chunks.
# #     """
# #     print(f"[INFO] Splitting text into chunks (chunk size: {chunk_size}, overlap: {chunk_overlap})...")
# #     chunks = []
# #     start = 0
# #     while start < len(text):
# #         end = start + chunk_size
# #         chunk = text[start:end]
# #         chunks.append(chunk)
# #         start += chunk_size - chunk_overlap

# #     print(f"[INFO] Total chunks created: {len(chunks)}")
# #     return chunks

# # def process_pdf_for_llm(pdf_path, chunk_size=500, chunk_overlap=50):
# #     """
# #     High-level function to extract and chunk PDF content.
# #     """
# #     text = extract_text_from_pdf(pdf_path)
# #     chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
# #     return chunks

# # # Example usage
# # if __name__ == "__main__":
# #     pdf_file = "./data/academics.pdf"  # Replace with your PDF path
# #     chunk_size = 1000
# #     chunk_overlap = 100

# #     if os.path.exists(pdf_file):
# #         print("[INFO] PDF file found. Processing...\n")
# #         chunks = process_pdf_for_llm(pdf_file, chunk_size, chunk_overlap)

# #         print("\n========== CHUNK PREVIEW ==========")
# #         preview_count = min(3, len(chunks))
# #         for i in range(preview_count):
# #             print(f"\n--- Chunk {i+1} ---\n{chunks[i][:300]}...")  # Preview first 300 characters
        
# #         if len(chunks) > preview_count:
# #             print(f"\n--- Last Chunk ({len(chunks)}) ---\n{chunks[-1][:300]}...")

# #         print("\n[INFO] Finished processing and chunking the PDF.")
# #     else:
# #         print(f"[ERROR] PDF file not found at path: {pdf_file}")

# from pypdf import PdfReader

# def extract_raw_text_from_pdf(pdf_path):
#     """Extracts raw text content from a PDF file.

#     Args:
#         pdf_path (str): The path to the PDF file.

#     Returns:
#         str: A string containing all the text content of the PDF.
#              Returns an empty string if the file cannot be opened or read.
#     """
#     text = ""
#     try:
#         with open(pdf_path, 'rb') as pdf_file:
#             pdf_reader = PdfReader(pdf_file)
#             for page_num in range(len(pdf_reader.pages)):
#                 page = pdf_reader.pages[page_num]
#                 text += page.extract_text() + "\n\n"  # Add page breaks for readability
#     except Exception as e:
#         print(f"Error reading PDF: {e}")
#     return text

# def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
#     """
#     Splits the extracted text into overlapping chunks.
#     """
#     print(f"[INFO] Splitting text into chunks (chunk size: {chunk_size}, overlap: {chunk_overlap})...")
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += chunk_size - chunk_overlap

#     print(f"[INFO] Total chunks created: {len(chunks)}")
#     return chunks

# def process_pdf_for_llm(pdf_path, chunk_size=500, chunk_overlap=50):
#     """
#     High-level function to extract and chunk PDF content.
#     """
#     text = extract_raw_text_from_pdf(pdf_path)
#     chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
#     return chunks

# if __name__ == "__main__":
#     pdf_file_path = "./data/finance.pdf"  # Replace with the actual path to your PDF
#     raw_text = extract_raw_text_from_pdf(pdf_file_path)
#     if raw_text:
#         # You can now feed this 'raw_text' to your LLM
#         print("Extracted Text:\n")
#         print(raw_text)  # Print the entire extracted text
#     else:
#         print("No text extracted from the PDF.")

from langchain_community.document_loaders import UnstructuredPDFLoader

def extract_text_with_unstructured(pdf_path):
    """Extracts text content from a PDF file using UnstructuredPDFLoader."""
    try:
        loader = UnstructuredPDFLoader(pdf_path)
        documents = loader.load()
        all_text = ""
        for doc in documents:
            all_text += doc.page_content + "\n\n"
        return all_text
    except Exception as e:
        print(f"Error reading PDF with UnstructuredPDFLoader: {e}")
        return ""

def recursive_character_split(text: str, chunk_size: int, chunk_overlap: int = 0, separators: list[str] = ["\n\n", "\n", " ", ""]) -> list[str]:
    """Recursively splits text into chunks based on a list of separators."""
    print(f"--- Starting custom chunking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap} ---")
    print(f"Initial text length: {len(text)}")

    chunks = [text]
    final_chunks = []

    for i, separator in enumerate(separators):
        print(f"\nAttempting split with separator: '{repr(separator)}' (index {i})")
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                print(f"  - Chunk of length {len(chunk)} is too large. Splitting...")
                split_texts = chunk.split(separator)
                print(f"    - Split into {len(split_texts)} parts.")
                for split_text in split_texts:
                    if split_text:
                        new_chunks.append(split_text)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
        print(f"  - After splitting with '{repr(separator)}', there are {len(chunks)} chunks.")

    print("\n--- Handling chunk size and overlap ---")
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"  - Processing chunk {i+1} of length {len(chunk)}: '{chunk[:20]}...'")
        if len(chunk) <= chunk_size:
            processed_chunks.append(chunk)
            print("    - Chunk is within size limit.")
        else:
            print("    - Chunk is still too large. Force splitting...")
            n = len(chunk)
            for j in range(0, n, max(1, chunk_size - chunk_overlap)):  # Avoid zero step
                sub_chunk = chunk[j : j + chunk_size]
                processed_chunks.append(sub_chunk)
                print(f"      - Forced sub-chunk of length {len(sub_chunk)}: '{sub_chunk[:20]}...'")

    print("\n--- Applying overlap ---")
    final_chunks = []
    for i, chunk in enumerate(processed_chunks):
        final_chunks.append(chunk)
        print(f"  - Added chunk {i+1} of length {len(chunk)}: '{chunk[:20]}...'")
        if chunk_overlap > 0 and i < len(processed_chunks) - 1:
            overlap = chunk[-chunk_overlap:]
            # Langchain's implementation typically prepends the overlap to the next chunk
            # or creates separate overlapping chunks. For simplicity, we're just noting it.
            print(f"    - Potential overlap from current chunk: '{overlap}'")

    print("\n--- Final Custom Chunks ---")
    for i, chunk in enumerate(final_chunks):
        print(f"  - Chunk {i+1}: \"{chunk}\" (Length: {len(chunk)})")
        print("-" * 40)

    return [chunk.strip() for chunk in final_chunks if chunk.strip()]

if __name__ == "__main__":
    pdf_file_path = "./data/academics.pdf"
    raw_text = extract_text_with_unstructured(pdf_file_path)

    if raw_text:
        print("Extracted Text (using UnstructuredPDFLoader):\n")
        # print(raw_text)  # Print the entire extracted text (commented out)

        chunks = recursive_character_split(
            raw_text,
            chunk_size=1500,  # You can adjust this value
            chunk_overlap=200,  # You can adjust this value
            separators=["\n\n", "\n", " ", ""],
        )

    else:
        print("No text extracted from the PDF using UnstructuredPDFLoader.")