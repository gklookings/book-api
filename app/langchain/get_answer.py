
import os.path
import bs4
from langchain_community.document_loaders import (
  Docx2txtLoader
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output.txt")
    ]
)

logger = logging.getLogger(__name__)

llm=ChatOpenAI(model="gpt-4o",temperature=0.0)

chunk_size =1000
chunk_overlap = 100

def get_answer(question):
  try:
    # Load, chunk and index the contents.
    loaders = [
       Docx2txtLoader("app/langchain/books/CrimePunishment.docx"),
       Docx2txtLoader("app/langchain/books/خطوة بخطوة ويدا بيد مساء 20 -2-2024 3 copy.docx"),
       Docx2txtLoader("app/langchain/books/الرِّحلة العياشيَّة 1661–1663م- المؤلف أبو سالم عبد الله بن محمد العياشي- المجلد الأول.docx"),
       Docx2txtLoader("app/langchain/books/الرِّحلة العياشيَّة 1661–1663م- المؤلف أبو سالم عبد الله بن محمد العياشي- المجلد الثاني.docx")
    ]
    
    # documents = loaders.load_and_split()
    documents = []
    for loader in loaders:
        documents.extend(loader.load_and_split())

    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
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

    retriever=vectorstore.as_retriever(search_kwargs={"k":75})

    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    answer = qa_chain.invoke({"input": question})

    print(answer)

    return answer
  
  except Exception as e:
      # Handle the exception gracefully
      logger.error(f"An error occurred: {e}", exc_info=True)
      return f"An error occured on answering. Please try again {e}"  # Or return an appropriate value indicating failure
