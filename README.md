# aws_bedrock_chatwithpdf
This project uses AWS Bedrock with Amazon Titan embeddings, Anthropic Claude LLM, LangChain, FAISS vector store, and Streamlit to build an AI-powered PDF question-answering app that extracts and summarizes information from PDF documents efficiently.
# AI-Powered PDF Question Answering with AWS Bedrock

This project is an AI-based application that allows users to ask questions from PDF documents. It uses AWS Bedrock with Amazon Titan embeddings and Anthropic Claude LLM, LangChain for chaining, FAISS for vector search, and Streamlit for the web interface.

## Features

- Ingest PDF files and split text for processing  
- Generate embeddings using Amazon Titan model  
- Store embeddings with FAISS for efficient similarity search  
- Use Anthropic Claude model via AWS Bedrock for detailed answers  
- Interactive UI with Streamlit to ask questions and get answers  

## Requirements

- Python 3.8+  
- AWS account with Bedrock access  
- Install dependencies from `requirements.txt`  

## Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/your-repo.git
