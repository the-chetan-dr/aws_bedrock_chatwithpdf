import json
import os
import sys
import boto3
import streamlit as st

#using amazon titan model fro embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


#data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.vectorstores import Faiss


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

#data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()


    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=10000)

    docs=text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    vectorstore_faiss=Faiss.from_documents(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
def load_vector_store():
    return Faiss.load_local("faiss_index", bedrock_embeddings)

    

def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock, model_kwargs={"max_tokens": 512})


    return llm

prompt_template = """Human:
Use the following pieces of context to provide a detailed answer to the question at the end. Your answer should be at least 250 words and include detailed explanations. 
If you don't know the answer, just say you don't know. Do not make up an answer.

<context>
{context}
</context>

Question: {question}
Assistant:"""

PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])

def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
        search_type="similarity",search_kwargs={"k":3}),
        return_source_documents=True,

        chain_type_kwargs={"prompt":PROMPT}
    )
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("chat pdf")
    st.header("chat with PDF using AWS bedrock")

    user_question=st.text_input("Ask a question from  PDF files")
    if user_question:
        with st.spinner("Getting your answer..."):
            vectorstore = load_vector_store()
            llm = get_claude_llm()
            response = get_response_llm(llm, vectorstore, user_question)
            st.write(response)

    with st.sidebar:
        st.title("Menu")

        if st.button("Vector Update"):
            with st.spinner("finding the best answer for you...."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")

if __name__=="__main__":
    main()              
    
        
    





