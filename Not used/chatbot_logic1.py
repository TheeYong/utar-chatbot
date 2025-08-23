from openai import AzureOpenAI
import os
import logging
from typing import List, Dict, Any
import json
import glob
import ollama

# LangChain imports for document processing and vector database
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# embedding_function = OpenAIEmbeddings(
#     azure_deployment="your-embedding-deployment-name",
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_API_KEY
# )

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_debug.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)

# Constants
# ollama.base_url = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"
# EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = "mxbai-embed-large"

# Define department-specific vector stores
VECTOR_STORES = {
    "Division of Admissions and Credit Evaluation": "./vector_db/admissions",
    "Division of Finance": "./vector_db/finance",
    "Department of Examination and Awards": "./vector_db/examinations",
}

# Define department descriptions
DEPARTMENTS = {
    "Division of Admissions and Credit Evaluation": "Division of Admissions and Credit Evaluation: Handles admissions-related queries.",
    "Division of Finance": "Division of Finance: Handles finance, fees, and scholarship queries.",
    "Department of Examination and Awards": "Department of Examination and Awards: Handles course and exam queries.",
}

def debug_print(message: str, level: str = "info"):
    """Enhanced debug printing with multiple logging levels."""
    print(f"[DEBUG] {message}")
    
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "debug":
        logging.debug(message)


def classify_department(query):
    """Use Ollama directly to determine the relevant department."""
    classification_prompt = f"""
    You are a university chatbot that routes queries to the correct department.

    Departments:
    - Division of Admissions and Credit Evaluation: Handles admissions-related queries.
    - Division of Finance: Handles finance, fees, and scholarship queries.
    - Department of Examination and Awards: Handles course and exam queries.

    Question: "{query}"

    Return ONLY one department which is one phrase only: Division of Admissions and Credit Evaluation, Division of Finance, or Department of Examination and Awards.
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": classification_prompt}]
        )
        
        department = response['message']['content'].strip()
        
        # Ensure response is one of the expected departments
        expected_departments = {"Division of Admissions and Credit Evaluation", "Division of Finance", "Department of Examination and Awards"}
        for dept in expected_departments:
            if dept in department:
                logging.info(f"Classified query as: {dept}")
                return dept
        
        logging.warning(f"Could not classify query correctly. Got: {department}")
        return None
    except Exception as e:
        logging.error(f"Error classifying department: {e}")
        return None

def process_query(query, department, vector_db):
    """Process a query for a specific department."""
    vector_database = get_vector_database(department)
    if vector_database is None:
        return "Department database not available."
    
    # Retrieve relevant documents
    contexts = retrieve_documents(vector_database, query)
    
    # Generate and stream the answer
    print("\nChatbot Response:")
    for chunk in answer_question(contexts, query, department):
        print(chunk, end="", flush=True)


def ingest_pdf(doc_folder_path):
    """Load all PDF documents in a folder using LangChain with enhanced logging."""
    debug_print(f"Looking for PDFs in: {doc_folder_path}")
    
    if not os.path.exists(doc_folder_path):
        debug_print(f"Folder not found: {doc_folder_path}", level="error")
        return None

    all_data = []
    pdf_files = glob.glob(os.path.join(doc_folder_path, "*.pdf"))

    if not pdf_files:
        debug_print(f"No PDF files found in: {doc_folder_path}", level="error")
        return None

    for pdf_file in pdf_files:
        try:
            debug_print(f"Loading PDF: {pdf_file}")
            loader = UnstructuredPDFLoader(file_path=pdf_file)
            data = loader.load()
            all_data.extend(data)
            debug_print(f"Loaded {len(data)} documents from {pdf_file}")
        except Exception as e:
            debug_print(f"Failed to load {pdf_file}: {e}", level="error")

    debug_print(f"Total documents loaded: {len(all_data)}")
    if all_data:
        debug_print(f"Preview (first 200 chars): {all_data[0].page_content[:200]}")

    return all_data


def split_documents(documents):
    """Split documents into smaller chunks with detailed logging."""
    debug_print("Starting document splitting process")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    debug_print(f"Total document chunks created: {len(chunks)}")
    debug_print("Sample of first 3 document chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        debug_print(f"Chunk {i} (length {len(chunk.page_content)}):\n{chunk.page_content[:300]}...\n")
    
    return chunks


def get_vector_database(department):
    """Load or create a vector database for a department using LangChain."""
    vector_path = VECTOR_STORES.get(department)
    
    if not vector_path:
        logging.error(f"Unknown department: {department}")
        return None

    try:
        if os.path.exists(vector_path):
            logging.info(f"Loading vector database for {department} from {vector_path}...")
            vector_database = Chroma(
                persist_directory=vector_path, 
                embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
            )
        else:
            logging.info(f"Creating vector database for {department}...")
            doc_folder_path = f"./data/{department}"
            data = ingest_pdf(doc_folder_path)
            if data is None:
                return None

            chunks = split_documents(data)
            ollama.pull(EMBEDDING_MODEL)

            vector_database = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
                persist_directory=vector_path
            )

        return vector_database
    
    except Exception as e:
        logging.error(f"Failed to load or create vector database for {department}: {e}")
        return None


# Replace LangChain retriever with direct calls to ollama
def generate_query_variations(question: str) -> List[str]:
    """Generate alternative versions of the question using Ollama directly."""
    prompt = f"""You are an AI chatbot for a university. 
    Generate 5 alternative versions of this question to improve document retrieval:
    
    Original question: {question}
    
    Return only the questions as a JSON array, like this: ["question1", "question2", "question3", "question4", "question5"]
    """
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {"role": "user", "content": prompt}
        ])
        
        # Try to parse the response as JSON
        content = response['message']['content']
        try:
            # Extract JSON array if response contains explanatory text
            if '[' in content and ']' in content:
                json_str = content[content.find('['):content.rfind(']')+1]
                variations = json.loads(json_str)
            else:
                variations = json.loads(content)
            
            # Add original question to variations
            variations.append(question)
            logging.info(f"Generated {len(variations)} query variations")
            return variations
        except json.JSONDecodeError:
            # If JSON parsing fails, split by newlines and filter out empty lines
            variations = [line.strip().strip('"-').strip() for line in content.split('\n') 
                         if line.strip() and not line.strip().startswith('[') and not line.strip().startswith(']')]
            variations.append(question)
            logging.info(f"Generated {len(variations)} query variations (fallback method)")
            return variations
    except Exception as e:
        logging.error(f"Error generating query variations: {e}")
        return [question]  # Return original question if generation fails



def retrieve_documents(vector_database, question: str, k: int = 3) -> List[str]:
    """Retrieve relevant documents with improved deduplication."""
    variations = generate_query_variations(question)
    
    # Get unique documents from all variations
    all_docs = []
    unique_doc_contents = set()
    
    for query in variations:
        docs = vector_database.similarity_search(query, k=k)
        
        # Filter out near-duplicate documents
        for doc in docs:
            # Use a hash or normalized content to detect near-duplicates
            normalized_content = ' '.join(doc.page_content.split()[:20])  # First 20 words
            
            if normalized_content not in unique_doc_contents:
                unique_doc_contents.add(normalized_content)
                all_docs.append(doc)
    
    # Limit to unique documents
    unique_docs = all_docs[:k*2]
    
    # Extract text content from documents
    contexts = [doc.page_content for doc in unique_docs]
    
    # Logging and debugging
    debug_print(f"Retrieved {len(contexts)} unique documents")
    for i, context in enumerate(contexts, 1):
        debug_print(f"Context {i} (first 300 chars): {context[:300]}...")
    
    return contexts


def answer_question(contexts: List[str], question: str, department: str) -> str:
    """Generate a more refined answer based on contexts."""
    # Combine and deduplicate contexts
    unique_contexts = list(dict.fromkeys(contexts))
    combined_context = "\n\n---\n\n".join(unique_contexts)
    
    prompt = f"""You are a university chatbot. Answer the question strictly based on the context, 
    avoiding redundant information and focusing on the most relevant details.

    Context:
    {combined_context}

    Question: {question}

    Provide a concise, clear and non-redundant response. If you are not sure of the answer or the answer cannot be found
    within the context, you just tell the user to refer to the {department} department.
    """
    # prompt = f"""You are a university chatbot. Answer the question in two steps:

    # Context:
    # {combined_context}

    # Question: {question}

    # Step 1: First analyze the context to identify key information and relationships between items. For example, if the context shows that certain items are components of others, note this explicitly.

    # Step 2: Based on your analysis, provide a concise and accurate answer that avoids redundancy and correctly represents relationships between items mentioned in the context. 

    # Provide a concise, clear and non-redundant response in a conversational tone, directly addressing the user. Avoid explicitly stating any steps or analysis. 
    # If you are not sure of the answer or the answer cannot be found within the context, tell the user to refer to the {department} department.
    # """
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        answer = ""
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                answer += chunk['message']['content']
                yield chunk['message']['content']
        
        return answer
    except Exception as e:
        error_msg = f"Error generating answer: {e}"
        logging.error(error_msg)
        return "An error occurred while generating the answer."
