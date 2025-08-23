# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
#     # other params...
# )


# from langchain_core.messages import AIMessage

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)
# print(ai_msg.content)

# main.py

import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/Examination_Instructions_Candidates_20201009.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_PATH = "./vector_db"
VECTOR_STORE_NAME = "knowledge-base"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


# def create_vector_database(chunks):
#     """Create a vector database from document chunks."""
#     # Pull the embedding model if not already available
#     ollama.pull(EMBEDDING_MODEL)

#     vector_database = Chroma.from_documents(
#         documents=chunks,
#         embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
#         collection_name=VECTOR_STORE_NAME,
#     )
#     logging.info("Vector database created.")
#     return vector_database

def get_vector_database():
    """Check if vector database exists and load or create it."""
    if os.path.exists(VECTOR_STORE_PATH):
        logging.info("Loading existing vector database...")
        vector_database = Chroma(
            persist_directory=VECTOR_STORE_PATH, 
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL))
        logging.info("Loaded existing vector database.")
    else:
        logging.info("Creating new vector database...")
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None
        
        chunks = split_documents(data)
        ollama.pull(EMBEDDING_MODEL)

        vector_database = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=VECTOR_STORE_PATH
        )
        logging.info("Vector database created and saved.")
    
    return vector_database


def create_retriever(vector_database, llm):
    """Create a multi-query retriever."""
    #Expands user's question into multiple queries
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant to answer the queries of university students. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    # Retrieve the most relevant documents from the vector database
    retriever = MultiQueryRetriever.from_llm(
        vector_database.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
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

    logging.info("Chain created successfully.")
    return chain


def main():
    # Create the vector database
    vector_database = get_vector_database()
    if vector_database is None:
        return

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_database, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = "what will happen if a candidate cheat during the final examination but is not caught by invigilators?"

    # # Get the response
    # res = chain.invoke(input=question)
    # print("Response:")
    # print(res)
    # Get the response in chunks
    print("New Response:")
    for chunk in chain.stream(input=question):
        print(chunk, end="", flush=True)  # Print without newline for real-time effect


if __name__ == "__main__":
    main()
