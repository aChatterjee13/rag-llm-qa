# RAG Pipeline

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [How to read and use the repo](#repo-usage)

## Introduction
Provides a Chat RAG pipeline using Langchain on a single PDF file

## Features
List the key features of the project:
- Uses Langchain
- Endpoints are routed
- Langserve and Langsmith for tracking and serving

## Installation
Describe the steps to install the project:
1. Clone the repository: `git clone https://github.com/yourusername/rag-project.git`
2. Navigate to the project directory: `cd rag-project`
3. Install the dependencies: pip install -r requirements.txt

## Contributing
Explain how others can contribute to the project. Include guidelines for submitting issues and pull requests.

## License
State the license under which the project is distributed. For example:
```
This project is licensed under the MIT License - see the LICENSE file for details.
```

## Contact
Provide contact information for the project maintainers:
- Name: Anindya Chatterjee
- Email: anindya.wbut@gmail.com
- GitHub: [aChatterjee13](https://github.com/aChatterjee13)

## repo-usage

conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt

Install Ollama
Ollama - Lets you run LLM locally. Provides a wrapper around open source LLMs
Download Ollama - https://ollama.com/download

Download 2 models to run locally.
ollama run llama3.2
ollama run llama2

Put the ConceptsofBiology.PDF in the data/ folder before any operation

There are notebooks to help with understanding how the langchain framework is used
1. Langchain API validation Checker
2. RAG with Ollama Llama3.2 and FAISS vectorstore
3. RAG with Huggingface Mistral 7B and FAISS vectorstore
4. RAG with GPT3.5 turbo and Chroma vector store
5. Multiple embedding models are tried like OpenAI, Ollama, HuggingFace etc as shown in the noebooks

> The congif/app.yaml files contain the api keys and tokens for OpenAI, LangChain and Hugging Face. DO REPLACE WITH YOUR PERSONAL API KEYS

1. app.py - Langchain server with the endpoint routes defined. Implementation for 3 LLMs are provided.
2. client.py - Streamlit app for basic QA on Biology

> How to run?
1. python app.py 
2. streamlit run client.py
