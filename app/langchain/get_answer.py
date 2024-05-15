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
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def get_answer(question):
  try:
    # Load, chunk and index the contents.
    loader = PyPDFLoader("app/langchain/books/crime-and-punishment.pdf")
    docs = loader.load_and_split()

    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
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
    retriever.add_documents(docs)

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

    return answer
  
  except Exception as e:
      # Handle the exception gracefully
      print(f"An error occurred in langchain: {e}")
      return "An error occured. Please try again"  # Or return an appropriate value indicating failure