
import os.path
import bs4
from pathlib import Path
from langchain import hub
from langchain_community.document_loaders import (
  PyPDFLoader,
  Docx2txtLoader
)
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.retrievers import (MultiQueryRetriever,ParentDocumentRetriever)
from langchain.storage import (InMemoryStore,LocalFileStore)
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

llm = ChatOpenAI(model="gpt-3.5-turbo")

def get_answer(question):
  try:
    # Load, chunk and index the contents.
    loader = Docx2txtLoader("app/langchain/books/CrimePunishment.docx")
    documents = loader.load_and_split()

      # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
          collection_name="split_parents", 
          embedding_function=OpenAIEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(documents)

    retrieved_docs = retriever.invoke(question)

    # doc_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
    # docs = doc_splitter.split_documents(documents)
    
    # PERSIST_DIR = "./chroma_db"
    # if not os.path.exists(PERSIST_DIR):
    #   embedding_function = OpenAIEmbeddings()
    #   vectorstore = Chroma.from_documents(
    #         docs,
    #         embedding_function,
    #         persist_directory=PERSIST_DIR
    #   )
    # else:
    #     vectorstore = Chroma(
    #       persist_directory=PERSIST_DIR,
    #       embedding_function=OpenAIEmbeddings(),
    #   )

    # retriever=vectorstore.as_retriever()
    # llm_model = ChatOpenAI(temperature=0)
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=retriever, llm=llm_model
    # )
    # unique_docs = retriever_from_llm.invoke(question)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    template = """Use the following pieces of context and the broader conext of the document or the story to answer the question at the end.
    Answer the question in detail.

    {context}

    Question: {question} """
    
    custom_rag_prompt = PromptTemplate.from_template(template)


    rag_chain = (
        {"context": retriever|format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    return {"answer":answer,"docs":retrieved_docs}
  
  except Exception as e:
      # Handle the exception gracefully
      logger.error(f"An error occurred: {e}", exc_info=True)
      return "An error occured. Please try again"  # Or return an appropriate value indicating failure
