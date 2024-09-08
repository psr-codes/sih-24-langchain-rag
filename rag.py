import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load OpenAI API key from environment variables
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OpenAI API key is not set in environment variables.")

os.environ["OPENAI_API_KEY"] = "api key paste here ya dotenv kuch krke dekh lo..."

# Function to load and extract text from the PDF


def load_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

# Function to split the text into smaller chunks


def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

# Function to perform document search


def query_documents(query):
    # Load the PDF from a folder
    file_path = "./public/data-src.pdf"  # Update the path to your PDF file
    raw_text = load_pdf(file_path)

    # Split the text into chunks
    texts = split_text(raw_text)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vectorstore from texts
    docsearch = FAISS.from_texts(texts, embeddings)

    # Perform similarity search
    docs = docsearch.similarity_search(query)

    # Initialize OpenAI model and chain for question-answering
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Get the response based on the input query
    response = chain.run(input_documents=docs, question=query)

    return response
