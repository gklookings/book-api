import os
import bs4
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging
from openai import OpenAI

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output.txt")
    ]
)

logger = logging.getLogger(__name__)

client = OpenAI()

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
loaders.append(docLists(Docx2txtLoader("app/langchain/books/motanabi-15-all.docx"),119))
loaders.append(docLists(PyPDFLoader("app/langchain/books/motanabi-timeline.pdf"),120))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/motanabi-poems-all.docx"),121))
loaders.append(docLists(PyPDFLoader("app/langchain/books/26-1.pdf"),122))


for i in range(1, 325):  # From 1 to 324 inclusive
    filename = f"Ex-mot{i}.txt"  # Construct the filename
    book_id = 123 + i - 1  # Compute the ID
    loaders.append(docLists(TextLoader(f"app/langchain/books/{filename}"), book_id))

#vector store for docs other than poems
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

#vector store for poems
def create_combined_vector_store():
    all_docs = []
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for loader in loaders:
        if 123 <= loader.id <= 324:  # Check if the id is within the range 123 to 324
            documents = loader.name.load_and_split()
            docs = doc_splitter.split_documents(documents)
            all_docs.extend(docs)


    PERSIST_DIR = "./chroma_db/combined"
    embedding_function = OpenAIEmbeddings()

    if not os.path.exists(PERSIST_DIR):
        vectorstore = Chroma.from_documents(
            all_docs,
            embedding_function,
            persist_directory=PERSIST_DIR
        )
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_function,
        )

    return vectorstore


#get answer function
def get_answer(question,bookId):
    try:
        if bookId in {127}:
            vectorstore = create_combined_vector_store()
        else:
            for loader in loaders:
                if bookId == loader.id:
                    vectorstore = create_vector_store_for_document(loader.name, bookId)
                    break
          
        retrievers = vectorstore.as_retriever(search_kwargs={"k": 350})

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
