import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from langchain_community.vectorstores import PGVector
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Initialize the components only once for efficiency
def initialize_qa():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    # llm = ChatGroq(model='llama-3.1-70b-versatile', temperature=0, api_key='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')
    embeddings = HuggingFaceEmbeddings()
    
    db = PGVector.from_existing_index(
        embedding=embeddings,
        connection_string='postgresql+psycopg2://aibook:evaibooks_12@ai-books-instance-1.cncnbuvqyldu.eu-central-1.rds.amazonaws.com/books',
        collection_name='poems_vector',
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 200})

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    return qa_chain

qa_chain = initialize_qa()

def get_answer(question: str):
    response = qa_chain.invoke(question)
    return response['result']
