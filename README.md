PDF Query Assistant
This project is a PDF Query Assistant that allows users to query information from a PDF document using natural language. The assistant uses various libraries and tools to read, process, and retrieve information from the PDF.

Prerequisites
Python 3.7 or higher
Required Python libraries:
langchain
langchain_community
PyPDF2
streamlit
Installation
Clone the repository:

git clone https://github.com/your-repo/pdf-query-assistant.git
cd pdf-query-assistant
Install the required libraries:

pip install langchain langchain_community PyPDF2 streamlit
Usage
Reading the PDF File: The code reads a PDF file and extracts its text content.

from PyPDF2 import PdfReader

doc = "C:\\Users\\asus\\OneDrive\\Desktop\\pdf\\Data Science Interview Questions and Answers.pdf"
raw_text = ""
pdf_reader = PdfReader(doc)
for page in pdf_reader.pages:
    raw_text += page.extract_text()
Creating Chunks of Data: The text is split into smaller chunks for easier processing.

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
text_chunks = text_splitter.split_text(raw_text)
Vector Embedding of Chunks: The text chunks are embedded using Spacy embeddings and stored in a FAISS vector store.

from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
vector_storage = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
vector_storage.save_local("faiss_db")
Retrieving Data: The vector store is loaded, and a retriever tool is created to fetch relevant information.

db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
from langchain.tools.retriever import create_retriever_tool
retrieval_chain = create_retriever_tool(retriever, "pdf_reader", "It is a tool to read data from pdfs")
Language Model and Prompt Template: A language model and prompt template are set up to generate responses based on user queries.

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key="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")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, \"answer is not available in the context\", don't provide the wrong answer"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
Creating and Executing the Agent: An agent is created and executed to handle user queries.

from langchain.agents import AgentExecutor, create_tool_calling_agent

tool = [retrieval_chain]
agent = create_tool_calling_agent(llm, tool, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
Streamlit Interface: A simple Streamlit interface is provided for user interaction.

import streamlit as st

st.title("PDF Query Assistant")
user_input = st.text_input("Enter your question:")

if st.button("Find Answer"):
    response = agent_executor.invoke({"input": user_input})
    st.write(f"Input: {response['input']}")
    st.write(f"Output: {response['output']}")
Running the Application
To run the application, use the following command:

streamlit run app.py
Replace app.py with the name of your Python script file.


This README file provides a comprehensive overview of the project, including installation instructions, usage details, and how to run the application.
