import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

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
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info(f"Loaded PDF: {doc_path}")
        return data
    else:
        logging.error(f"PDF file not found: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def get_vector_database(department):
    """Load or create a vector database for a department."""
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


def create_retriever(vector_database, llm):
    """Create a multi-query retriever for document retrieval."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI chatbot for a university. 
        Generate 5 alternative versions of this question to improve document retrieval:
        
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_database.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create a query processing chain."""
    template = """Answer the question using only the following context:
    
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created.")
    return chain


def create_agent(department, llm):
    """Create an agent for a department."""
    vector_database = get_vector_database(department)
    if vector_database is None:
        return None

    retriever = create_retriever(vector_database, llm)
    chain = create_chain(retriever, llm)

    return {
        "name": department,
        "description": DEPARTMENTS[department],
        "chain": chain
    }


def classify_department(query, llm):
    """Use LLM to determine the relevant department."""
    classification_prompt = f"""
    You are a university chatbot that routes queries to the correct department.

    Departments:
    - Admissions: Handles admissions-related queries.
    - Finance: Handles finance, fees, and scholarship queries.
    - Academics: Handles course and exam queries.

    Question: "{query}"

    Return ONLY one word: Admissions, Finance, or Academics.
    """

    response = llm.invoke(classification_prompt)
    department = response.content.strip().lower()  # Normalize response

    # Ensure response is one of the expected departments
    expected_departments = {"admissions", "finance", "academics"}
    if department not in expected_departments:
        logging.warning("Could not classify query correctly.")
        return None

    logging.info(f"Classified query as: {department.capitalize()}")
    return department


def main():
    """Main chatbot execution."""
    llm = ChatOllama(model=MODEL_NAME)

    # Create department agents
    agents = {dept: create_agent(dept, llm) for dept in VECTOR_STORES.keys()}

    if None in agents.values():
        logging.error("One or more agents failed to initialize.")
        return

    # User query
    query = input("Ask your question: ")

    # Classify query
    department = classify_department(query, llm)

    if department and department in agents:
        logging.info(f"Routing query to {department} agent...")
        agent = agents[department]
        chain = agent["chain"]

        # Get response
        print("\nChatbot Response:")
        for chunk in chain.stream(input=query):
            print(chunk, end="", flush=True)
    else:
        print("\nSorry, I couldn't determine which department can answer your question.")


if __name__ == "__main__":
    main()
