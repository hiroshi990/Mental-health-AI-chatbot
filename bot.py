from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema import SystemMessage,HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.chains import StuffDocumentsChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import json
import pandas as pd


json_file_path="E:/rag_chatbot/intents.json"
db_folder_path="E:/rag_chatbot/db"

def create_vector_db(json_file_path,db_directory):
    loader=JSONLoader(
        file_path=json_file_path,
        jq_schema=".intents[]",
        text_content=False
    )
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(separators=["\n","."],
                                                chunk_size=1000,
                                                chunk_overlap=200)
    chunks=text_splitter.split_documents(documents=docs)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db=FAISS.from_documents(
        chunks,embeddings
    )

    vector_db.save_local(folder_path=db_directory)
    
create_vector_db(json_file_path,db_folder_path)
    
