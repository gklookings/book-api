
import os.path
import bs4
from langchain import hub
from langchain_community.document_loaders import (
  PyPDFLoader
)
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output.txt")
    ]
)

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def get_answer(question):
  try:
    # Load, chunk and index the contents.
    loader = PyPDFLoader("app/langchain/books/crime-and-punishment.pdf")
    documents = loader.load_and_split()

    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
    docs = doc_splitter.split_documents(documents)
    
    PERSIST_DIR = "./chroma_db"
    if not os.path.exists(PERSIST_DIR):
      embedding_function = OpenAIEmbeddings()
      vectorstore = Chroma.from_documents(
            docs,
            embedding_function,
            persist_directory=PERSIST_DIR
      )
    else:
        vectorstore = Chroma(
          persist_directory=PERSIST_DIR,
          embedding_function=OpenAIEmbeddings(),
      )

    retriever=vectorstore.as_retriever()
    llm_model = ChatOpenAI(temperature=0)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm_model
    )
    unique_docs = retriever_from_llm.invoke(question)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    template = """Use the following pieces of context and the broader conext of the document or the story to answer the question at the end.
    Answer the question in detail.

    {context}

    Question: {question} """
    
    custom_rag_prompt = PromptTemplate.from_template(template)


    rag_chain = (
        {"context": retriever_from_llm|format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    return {"answer":answer,"docs":unique_docs}
  
  except Exception as e:
      # Handle the exception gracefully
      logger.error(f"An error occurred: {e}", exc_info=True)
      return "An error occured. Please try again"  # Or return an appropriate value indicating failure
