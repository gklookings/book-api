from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from typing import Union
import requests
import tempfile
import os


def extract_text_from_file(file: Union[UploadFile, str]):
    try:
        if isinstance(file, str):
            response = requests.get(file)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail="Failed to download file from URL"
                )
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            print(f"Extracting text from the file located at: {temp_file_path}")
            _, file_extension = os.path.splitext(file)
            loader = None
            if file_extension.lower() == ".docx":
                loader = Docx2txtLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
            if loader:
                documents = loader.load()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

        else:
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".docx", ".pdf", ".txt", ".xlsx"]:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(file.file.read())
                temp_file_path = temp_file.name

            print(f"Extracting text from the file located at: {temp_file_path}")
            loader = None
            if file_extension.lower() == ".docx":
                loader = Docx2txtLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path=temp_file_path)
            elif file_extension.lower() == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=temp_file_path)
            if loader:
                documents = loader.load()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

        print(f"Text extracted from the file: {len(documents)} documents")

        return documents

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Failed to extract text from the file: {e}")
        return None
