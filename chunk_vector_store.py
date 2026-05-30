# imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import chroma
from langchain_huggingface import HuggingFaceEmbeddings

# load the HF_TOKEN
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

class ChunkVectorStore:
    def __init__(self):
        self.embedding_model=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})
    # just like doing doc splitting normally without this class implementation
    def split_into_chunks(self,file_path:str):
        # load doc
        doc=PyPDFLoader(file_path).load()
        # split doc
        # define text splitter
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=20)
        chunks=text_splitter.split_documents(doc)

        # get only the content without metadata 
        chunks=filter_complex_metadata(chunks)

        return chunks

    def store_to_vector_database(self,chunks):
        return chroma.Chroma.from_documents(documents=chunks,embedding=self.embedding_model)
