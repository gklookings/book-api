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
loaders.append(docLists(Docx2txtLoader("app/langchain/books/خطوة بخطوة ويدا بيد مساء 20 -2-2024 3 copy.docx"),101))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/الرِّحلة العياشيَّة 1661–1663م- المؤلف أبو سالم عبد الله بن محمد العياشي- المجلد الأول.docx"),102))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/الرِّحلة العياشيَّة 1661–1663م- المؤلف أبو سالم عبد الله بن محمد العياشي- المجلد الثاني.docx"),103))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/الحواشي- ليوناردو.docx"),104))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/االكتاب كامل - عربي - إكسبريس باتاغونيا.docx"),105))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/التاريخ الملحمي للإيطاليين وطعامهم.docx"),106))
loaders.append(docLists(PyPDFLoader("app/langchain/books/الراين- فيكتور هوجو- افصول المترجمة-2.pdf"),107))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/باريس - تأليف جوليان غرين ترجمة أميمة قاسم-.docx"),108))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/بورخيس في صقلية مع مرشد أعمى.docx"),109))
loaders.append(docLists(PyPDFLoader("app/langchain/books/بازار السكك الحديدية الكبير- بول ثيرو--.pdf"),110))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/حول العالم في ثمانين يوما--كامل .docx"),111))
loaders.append(docLists(PyPDFLoader("app/langchain/books/ساردينيا والبحر- دي اتش لورانس- - الكتاب كامل.pdf"),112))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/قرناء البروج_بيرني آشمان.docx"),113))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/قمم غير موطوءة وأودية غير مطروقة-.docx"),114))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/كروم سان لورنزو-.docx"),115))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/لا تثق في طاهٍ إيطاليٍ نحيل- كامل .docx"),116))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/ليوناردو دافنشي-رحلات العقل.docx"),117))
loaders.append(docLists(Docx2txtLoader("app/langchain/books/مع الأتراك في فلسطين_.docx"),118))

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
