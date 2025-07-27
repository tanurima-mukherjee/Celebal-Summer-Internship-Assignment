import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def process_file_and_create_vectorstore(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def get_response_from_query(query, vectorstore):
    llm = Groq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192")

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa_chain.run(query)
    return response
