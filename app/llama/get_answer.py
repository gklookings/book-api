import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
import logging
import sys
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
    HyDEQueryTransform
)
from llama_index.core.query_engine import (
    MultiStepQueryEngine,
    TransformQueryEngine
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_answer(question):
    llm = OpenAI(system_prompt="Always reply in the "+question+" language")
    # gpt4 = OpenAI(temperature=0, model="gpt-4")
    # llm=gpt4
    # set a global llm
    Settings.llm = llm
    # Check if storage already exists
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # Load the documents and create the index
        documents = SimpleDirectoryReader("app/llama/books").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # Store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)


    # Query the index
    # query_engine = index.as_query_engine(similarity_top_k=5)
    query_engine = index.as_query_engine()

    # Hyde
    # hyde = HyDEQueryTransform(include_original=True)
    # query_engine = index.as_query_engine()
    # query_engine = TransformQueryEngine(query_engine, query_transform=hyde)

    # MultiStep

    # index_summary = ""
    # step_decompose_transform = StepDecomposeQueryTransform(llm,verbose=True)
    # query_engine = index.as_query_engine(llm)
    # query_engine = MultiStepQueryEngine(
    #     query_engine=query_engine,
    #     query_transform=step_decompose_transform,
    #     index_summary=index_summary,
    # )

    response = query_engine.query(question)
    return response