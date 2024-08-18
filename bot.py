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
from dotenv import load_dotenv
load_dotenv()
import json
import pandas as pd


json_file_path="E:/rag_chatbot/intents.json"
db_folder_path="E:/rag_chatbot/db"

#function to create vector database
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
    
#creating custom prompt template for chatbot to be more specific about its task
custom_prompt=(
    """
    You are mental health Therapist AI chatbot who understands the
    intent behind the user's message based on the context provide below. Talk to the user
    and make them feel better and comfortable.Use the coontext below to understand the intent behind 
    the user message.In the context the tag key represents the intent behind the user message 
    and the pattern key represents the user message.Use the responses key to generate responses.

    Context:{context}
    user input:{query}
    Dont hallucinate, just say I dont know, I only answer queries related to mental health
    """)


prompt=PromptTemplate(
    template=custom_prompt,
    input_variables=["context","query"]
)
#function to define the llm 
def load_llm():
    llm=GoogleGenerativeAI(
        model="gemini-pro",  # using the open source gemini-pro model
        google_api_key=os.environ["GOOGLE_API_KEY"],
        max_output_tokens=100,
        temperature=0.7)
    return llm


# function to create the retrievel chain
def retrievel_qa(llm,db):
    retriever = db.as_retriever()
    rag_chain = (
    {"context": retriever,  "query": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
    )
    return rag_chain


#function to return the retrievel chain
def bot():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db=FAISS.load_local(db_folder_path,embeddings,allow_dangerous_deserialization=True)
    llm=load_llm()
    retrievel_chain=retrievel_qa(llm,vector_db)

    return retrievel_chain


# function to get the response from the chatbot
def response(query):
    result=bot().invoke
    responses=result(query)
    return responses

