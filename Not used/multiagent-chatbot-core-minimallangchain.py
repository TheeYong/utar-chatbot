import os
import logging
import ollama
from typing import List, Dict, Any
import json

# LangChain imports for document processing and vector database
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "qwen2.5:7b"
# EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = "mxbai-embed-large"

# Define department-specific vector stores
VECTOR_STORES = {
    "admissions": "./vector_db/admissions",
    "finance": "./vector_db/finance",
    "academics": "./vector_db/academics",
}

# Define department descriptions
DEPARTMENTS = {
    "admissions": "Admissions Department: Handles admissions-related queries.",
    "finance": "Finance Department: Handles finance, fees, and scholarship queries.",
    "academics": "Academics Department: Handles course and exam queries.",
}


def ingest_pdf(doc_path):
    """Load PDF documents using LangChain."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info(f"Loaded PDF: {doc_path}")
        return data
    else:
        logging.error(f"PDF file not found: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks using LangChain."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def get_vector_database(department):
    """Load or create a vector database for a department using LangChain."""
    vector_path = VECTOR_STORES.get(department)
    
    if not vector_path:
        logging.error(f"Unknown department: {department}")
        return None

    if os.path.exists(vector_path):
        logging.info(f"Loading vector database for {department}...")
        vector_database = Chroma(
            persist_directory=vector_path, 
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
        )
    else:
        logging.info(f"Creating vector database for {department}...")
        doc_path = f"./data/{department}.pdf"
        data = ingest_pdf(doc_path)
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
    """Retrieve relevant documents using multiple query variations."""
    variations = generate_query_variations(question)

    # Print out the generated query variations
    print("\nGenerated Query Variations:")
    for i, query in enumerate(variations, start=1):
        print(f"{i}. {query}")
    
    # Get unique documents from all variations
    all_docs = []
    for query in variations:
        docs = vector_database.similarity_search(query, k=k)
        all_docs.extend(docs)
    
    # Remove duplicates (by document content)
    unique_docs = []
    doc_contents = set()
    
    for doc in all_docs:
        if doc.page_content not in doc_contents:
            doc_contents.add(doc.page_content)
            unique_docs.append(doc)
    
    # Limit to top k*2 documents to avoid too many results
    unique_docs = unique_docs[:k*2] if len(unique_docs) > k*2 else unique_docs
    
    # Extract text content from documents
    contexts = [doc.page_content for doc in unique_docs]
    logging.info(f"Retrieved {len(contexts)} unique documents")
    
    return contexts


def answer_question(contexts: List[str], question: str) -> str:
    """Generate an answer based on retrieved contexts using Ollama directly."""
    # Combine contexts into a single text with separators
    combined_context = "\n\n---\n\n".join(contexts)
    
    prompt = f"""You are a university chatbot. Answer the question **strictly based** on the context below.
    
    If the context does not contain relevant information, simply say: "I do not have enough information to answer that."
    
    Context:
    {combined_context}

    Question: {question}

    Provide a concise and accurate response.
    """
    
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


def classify_department(query: str) -> str:
    """Use Ollama directly to determine the relevant department."""
    classification_prompt = f"""
    You are a university chatbot that routes queries to the correct department.

    Departments:
    - Admissions: Handles admissions-related queries.
    - Finance: Handles finance, fees, and scholarship queries.
    - Academics: Handles course and exam queries.

    Question: "{query}"

    Return ONLY one word: Admissions, Finance, or Academics.
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": classification_prompt}]
        )
        
        department = response['message']['content'].strip().lower()
        
        # Ensure response is one of the expected departments
        expected_departments = {"admissions", "finance", "academics"}
        for dept in expected_departments:
            if dept in department:
                logging.info(f"Classified query as: {dept}")
                return dept
        
        logging.warning(f"Could not classify query correctly. Got: {department}")
        return None
    except Exception as e:
        logging.error(f"Error classifying department: {e}")
        return None


def process_query(department: str, query: str):
    """Process a query for a specific department."""
    vector_database = get_vector_database(department)
    if vector_database is None:
        return "Department database not available."
    
    # Retrieve relevant documents
    contexts = retrieve_documents(vector_database, query)
    
    # Generate and stream the answer
    print("\nChatbot Response:")
    for chunk in answer_question(contexts, query):
        print(chunk, end="", flush=True)


def main():
    """Main chatbot execution."""
    # User query
    query = input("Ask your question: ")

    # Classify query
    department = classify_department(query)

    if department and department in VECTOR_STORES:
        logging.info(f"Routing query to {department} agent...")
        process_query(department, query)
    else:
        print("\nSorry, I couldn't determine which department can answer your question.")


if __name__ == "__main__":
    main()