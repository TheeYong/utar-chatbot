import os
import logging
import ollama
from typing import List, Dict, Any
import json
import glob

# LangChain imports for document processing and vector database
from openai import AzureOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
# from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ====================== Azure OpenAI Setup ======================
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

azure_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_API_ENDPOINT,
    api_key=AZURE_API_KEY,
)

# EMBEDDING_FUNCTION = AzureOpenAIEmbeddings(
#     azure_deployment="your-embedding-deployment-name",
#     azure_endpoint=AZURE_API_ENDPOINT,
#     api_key=AZURE_API_KEY
# )

EMBEDDING_MODEL = "mxbai-embed-large"


# ====================== Logging Setup ======================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chatbot_debug.log"), logging.StreamHandler()]
)



# ====================== Constants ======================
VECTOR_STORES = {
    "Division of Admissions and Credit Evaluation": "./vector_db/admissions",
    "Division of Finance": "./vector_db/finance",
    "Department of Examination and Awards": "./vector_db/examinations",
}

DEPARTMENTS = {
    "Division of Admissions and Credit Evaluation": "Handles admissions-related queries.",
    "Division of Finance": "Handles finance, fees, and scholarship queries.",
    "Department of Examination and Awards": "Handles course and exam queries.",
}



# ====================== Utility ======================
def debug_print(message: str, level: str = "info"):
    print(f"[DEBUG] {message}")
    getattr(logging, level)(message)



# ====================== Department Classification ======================
def classify_department(query):
    prompt = f"""
You are a university chatbot that routes queries to the correct department.

Departments:
- Division of Admissions and Credit Evaluation: Handles admissions-related queries.
- Division of Finance: Handles finance, fees, and scholarship queries.
- Department of Examination and Awards: Handles course and exam queries.

Question: "{query}"

Return ONLY one department: Division of Admissions and Credit Evaluation, Division of Finance, or Department of Examination and Awards.
"""
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        department = response.choices[0].message.content.strip()
        expected_departments = set(DEPARTMENTS.keys())

        for dept in expected_departments:
            if dept in department:
                return dept

        logging.warning(f"Unrecognized department: {department}")
        return None
    except Exception as e:
        logging.error(f"Error classifying department: {e}")
        return None


# ====================== Document Ingestion ======================
def ingest_pdf(doc_folder_path):
    debug_print(f"Looking for PDFs in: {doc_folder_path}")
    if not os.path.exists(doc_folder_path):
        debug_print(f"Folder not found: {doc_folder_path}", level="error")
        return None

    all_data = []
    for pdf_file in glob.glob(os.path.join(doc_folder_path, "*.pdf")):
        try:
            debug_print(f"Loading PDF: {pdf_file}")
            loader = UnstructuredPDFLoader(file_path=pdf_file)
            data = loader.load()
            all_data.extend(data)
        except Exception as e:
            debug_print(f"Failed to load {pdf_file}: {e}", level="error")

    return all_data


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# ====================== Vector DB ======================
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

# ====================== Query Variants ======================
def generate_query_variations(question: str) -> List[str]:
    prompt = f"""You are an AI assistant. Generate 5 alternative versions of the question to improve document search.
    Original: {question}

    Return the alternatives as a JSON list like: ["question1", "question2", ...]
    """

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        json_str = content[content.find('['):content.rfind(']')+1]
        variations = json.loads(json_str)
        variations.append(question)
        return variations
    except Exception as e:
        logging.error(f"Query variation error: {e}")
        return [question]


# ====================== Document Retrieval ======================
def retrieve_documents(vector_db, question: str, k: int = 3) -> List[str]:
    variations = generate_query_variations(question)
    all_docs, seen = [], set()

    for query in variations:
        docs = vector_db.similarity_search(query, k=k)
        for doc in docs:
            key = ' '.join(doc.page_content.split()[:20])
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    return [doc.page_content for doc in all_docs[:k * 2]]


# ====================== Answer Generation ======================
def answer_question(contexts: List[str], question: str, department: str):
    combined_context = "\n\n---\n\n".join(dict.fromkeys(contexts))
    prompt = f"""You are a university chatbot. Use the context to answer concisely.

Context:
{combined_context}

Question: {question}

Only answer based on the context. If you can't find an answer, say to refer to the {department}.
"""

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful university chatbot."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        answer = ""
        # for chunk in response:
        #     if hasattr(chunk.choices[0].delta, "content"):
        #         token = chunk.choices[0].delta.content
        #         answer += token
        #         yield token
        # return answer
        for chunk in response:
            # Safer handling of chunk structure
            try:
                if (hasattr(chunk, 'choices') and 
                    len(chunk.choices) > 0 and 
                    hasattr(chunk.choices[0], 'delta') and 
                    hasattr(chunk.choices[0].delta, 'content') and 
                    chunk.choices[0].delta.content is not None):
                    
                    token = chunk.choices[0].delta.content
                    answer += token
                    yield token
            except (AttributeError, IndexError) as e:
                logging.debug(f"Skipping chunk due to {str(e)}: {chunk}")
                continue
                
        # If no content was generated, yield a default message
        if not answer:
            default_message = f"I don't have enough information to answer that. Please contact the {department} directly."
            yield default_message
    except Exception as e:
        logging.error(f"Answer generation error: {e}")
        yield "An error occurred while generating the answer."

# ====================== Query Execution ======================
def process_query(query, department, vector_db):
    db = get_vector_database(department)
    if db is None:
        return "Department database not available."

    contexts = retrieve_documents(db, query)
    print("\nChatbot Response:")
    for chunk in answer_question(contexts, query, department):
        print(chunk, end="", flush=True)