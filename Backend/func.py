import os
import pickle
import fitz
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from duckduckgo_search import DDGS

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Constants
PDF_PATH = 'uploads/'
FAISS_FILE = 'faiss_index.index'
DOCSTORE_FILE = 'in_memory_docstore.pkl'
INDEX_TO_DOCSTORE_ID_FILE = 'index_to_docstore_id.pkl'

# Load LLM and embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_texts_from_folder(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            texts[filename] = extract_text_from_pdf(path)
    return texts

def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def prepare_chunks_with_metadata(texts_dict):
    chunks_with_meta = []
    for filename, text in texts_dict.items():
        chunks = chunk_text(text)
        chunks_with_meta.extend([{"text": chunk, "source": filename} for chunk in chunks])
    return chunks_with_meta

def store_embeddings(chunks_with_metadata):
    chunk_texts = [chunk["text"] for chunk in chunks_with_metadata]
    chunk_embeddings = embedding_model.embed_documents(chunk_texts)
    chunk_embeddings_np = np.array(chunk_embeddings).astype("float32")

    index = faiss.IndexFlatL2(chunk_embeddings_np.shape[1])
    index.add(chunk_embeddings_np)

    documents = [
        Document(page_content=chunk["text"], metadata={"source": chunk["source"]})
        for chunk in chunks_with_metadata
    ]
    docstore = InMemoryDocstore({i: documents[i] for i in range(len(documents))})
    index_to_docstore_id = {i: i for i in range(len(documents))}

    faiss.write_index(index, FAISS_FILE)
    with open(DOCSTORE_FILE, 'wb') as f:
        pickle.dump(docstore, f)
    with open(INDEX_TO_DOCSTORE_ID_FILE, 'wb') as f:
        pickle.dump(index_to_docstore_id, f)

    return FAISS(embedding_function=embedding_model, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

def load_embeddings():
    index = faiss.read_index(FAISS_FILE)
    with open(DOCSTORE_FILE, 'rb') as f:
        docstore = pickle.load(f)
    with open(INDEX_TO_DOCSTORE_ID_FILE, 'rb') as f:
        index_to_docstore_id = pickle.load(f)
    return FAISS(embedding_function=embedding_model, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

def initialize_vectorstore(use_existing_embeddings=True):
    if use_existing_embeddings:
        return load_embeddings()
    else:
        texts = extract_texts_from_folder(PDF_PATH)
        meta_chunks = prepare_chunks_with_metadata(texts)
        return store_embeddings(meta_chunks)

# Functional Utilities
def retrieve_relevant_chunks(query, vector_store, k=5):
    return vector_store.similarity_search(query, k=k)

def rag_query(query: str, vector_store, memory_instance):
    docs = retrieve_relevant_chunks(query, vector_store)
    context = " ".join([doc.page_content for doc in docs])
    history_text = "\n".join([msg.content for msg in memory_instance.chat_memory.messages if isinstance(msg, (HumanMessage, AIMessage)) and hasattr(msg, "content")])
    if not history_text:
        history_text = "No previous conversation context."
    prompt = f"""Use the previous conversation and document content to answer the question.

                Previous Conversation:
                {history_text}

                Document Context:
                {context}

                Question:
                {query}
            """
    return chat_model.invoke(prompt).content

def is_answerable(query: str, vector_store, memory_instance):
    docs = retrieve_relevant_chunks(query, vector_store)
    context = " ".join([doc.page_content for doc in docs])
    history_text = "\n".join([msg.content for msg in memory_instance.chat_memory.messages if isinstance(msg, (HumanMessage, AIMessage)) and hasattr(msg, "content")])
    if not history_text:
        history_text = "No previous conversation context."
    prompt = f"""Given the previous conversation and the extracted context from documents,
                can the following question be answered using the available information?

                Previous Conversation:
                {history_text}

                Document Context:
                {context}

                Question:
                {query}

                Answer only Yes or No.
            """
    result = chat_model.invoke(prompt).content.lower()
    return "Yes" if "yes" in result else "No"

def web_search(query: str):
    results = DDGS().text(query, max_results=5)
    info = "\n\n".join(doc["body"] for doc in results)
    prompt = f"Answer the query {query} based on the online resource information: {info}"
    return chat_model.invoke(prompt).content

def extract_keywords(query: str, vector_store):
    docs = retrieve_relevant_chunks(query, vector_store)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Extract key topics and keywords from this content:\n{context}"
    return chat_model.invoke(prompt).content

def summarize_text(query: str, vector_store, memory_instance):
    keywords_result = extract_keywords(query, vector_store)
    keywords = keywords_result.split(", ")[:10]
    history_text = "\n".join([msg.content for msg in memory_instance.chat_memory.messages if isinstance(msg, (HumanMessage, AIMessage)) and hasattr(msg, "content")])
    if not history_text:
        history_text = "No previous conversation context."
    prompt = f"""Using the previous conversation and the important keywords, provide a meaningful summary.

                Previous Conversation:
                {history_text}

                Focus Keywords: {', '.join(keywords)}

                Now, summarize the document accordingly.
            """
    return chat_model.invoke(prompt).content

def generate_mcqs(query: str, vector_store):
    docs = retrieve_relevant_chunks(query, vector_store)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Generate 5 MCQs from the following content:\n{context}"
    return chat_model.invoke(prompt).content

def mcq_feedback(input_text: str):
    prompt = f"Evaluate the following MCQ attempt:\n{input_text}\nProvide feedback."
    return chat_model.invoke(prompt).content