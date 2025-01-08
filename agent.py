
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_session_id
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Initialize the document loader
file_path = "/Users/jaydaksharora/Downloads/SRB-2023-24_compressed.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Configure Gemini Embeddings
genai.configure(api_key="AIzaSyB3YqnrS9jBzDA8EuASkd6gDNI8UQkTQJw")

class GeminiEmbeddings:
    def __init__(self, model_name="models/text-embedding-004", api_key="AIzaSyB3YqnrS9jBzDA8EuASkd6gDNI8UQkTQJw"):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_query(self, text):
        result = genai.embed_content(model=self.model_name, content=text)
        return result['embedding']

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

embeddings = GeminiEmbeddings()

# Split and vectorize the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory="./chroma_db")

# Export retriever object for reuse
retriever = vectorstore.as_retriever()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile",api_key="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")

# Create the prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Answer in detail and concise. "
    "You are a student resource book expert providing information about various rules of college."
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Set up the retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = rag_chain.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['answer']