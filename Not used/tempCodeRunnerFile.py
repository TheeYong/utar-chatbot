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
    # Load and process the PDF document
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = "How to report BOI?"

    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()
