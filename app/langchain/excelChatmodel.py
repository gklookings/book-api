import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Initialize the components only once for efficiency
def initialize_qa():
    llm = ChatGroq(model=os.getenv("LLM_MODEL_NAME"), temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    embeddings = HuggingFaceEmbeddings()
    os.environ["COLLECTION_NAME_EXCEL"] = "chatwithExcel"
    db = PGVector.from_existing_index(
        embedding=embeddings,
        connection_string=os.getenv("CONNECTION_STRING"),
        collection_name=os.getenv("COLLECTION_NAME_EXCEL"),
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    prompt_template = """
    You are a Conversational AI assistant that provides responses based on the given context. 
    If the answer is not present in the provided context, please respond with:
    "I'm sorry, but I don't have that information in the current context."
    
    Context:
    {context}
    
    Question:
    {question}
    """

    chat_prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    qa_excel_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    return qa_excel_chain

qa_excel_chain = initialize_qa()

def get_excel_response(question: str):
    response = qa_excel_chain.invoke(question)
    return response['result']
