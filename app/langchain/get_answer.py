import os
import bs4
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
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

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

chunk_size = 1000
chunk_overlap = 100
start_id = 100

class docLists:
    def __init__(self, name, id):
        self.name = name
        self.id = id

loaders=[]

loaders.append(docLists(Docx2txtLoader("app/langchain/books/CrimePunishment.docx"),100))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Italian Journey.docx"),101))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/The Ayyashi Journey 1661–1663 AD Volume 1.docx"),102))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/The Ayyashi Journey 1661–1663 AD Volume 2.docx"),103))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/footnotes.docx"),104))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/The Old Patagonian Express.docx"),105))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Delizia: Epic History of the Italians and their Food.docx"),106))
loaders.append(docLists(PyPDFLoader("app/langchain/books/THE RHINE.pdf"),107))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Paris.docx"),108))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Borges in Sicily: Journey with a Blind Guide.docx"),109))
loaders.append(docLists(PyPDFLoader("app/langchain/books/The Great Railway Bazaar.pdf"),110))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Around the World in Eighty Days.docx"),111))
loaders.append(docLists(PyPDFLoader("app/langchain/books/Sea and Sardinia.pdf"),112))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Sign Mates: Understanding the Games People Play.docx"),113))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Untrodden Peaks and Unfrequented Valleys.docx"),114))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/The Vines of San Lorenzo.docx"),115))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Never Trust A Skinny Italian Chef.docx"),116))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/Leonardo da Vinci Flights of the Mind.docx"),117))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/With the Turks in Palestine.docx"),118))

def create_vector_store_for_document(loader, doc_id):
    documents = loader.load_and_split()
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = doc_splitter.split_documents(documents)

    PERSIST_DIR = f"./chroma_db/{doc_id}"
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

    return vectorstore

def get_answer(question,bookId):
    try:
        for loader in loaders:
          if bookId == loader.id:
            vectorstore = create_vector_store_for_document(loader.name, bookId)
          
        retrievers = vectorstore.as_retriever(search_kwargs={"k": 75})

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

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retrievers, question_answer_chain)

        answer = qa_chain.invoke({"input": question})

        return answer

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return f"An error occurred on answering. Please try again {e}"
