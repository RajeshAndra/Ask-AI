import os
import pickle
import fitz
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

PDF_PATH = 'uploads/'
FAISS_FILE = 'faiss_index.index'
DOCSTORE_FILE = 'in_memory_docstore.pkl'
INDEX_TO_DOCSTORE_ID_FILE = 'index_to_docstore_id.pkl'

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config
)

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

    return FAISS(embedding_function=embedding_model, index=index,
                 docstore=docstore, index_to_docstore_id=index_to_docstore_id)

def load_embeddings():
    index = faiss.read_index(FAISS_FILE)
    with open(DOCSTORE_FILE, 'rb') as f:
        docstore = pickle.load(f)
    with open(INDEX_TO_DOCSTORE_ID_FILE, 'rb') as f:
        index_to_docstore_id = pickle.load(f)

    return FAISS(embedding_function=embedding_model, index=index,
                 docstore=docstore, index_to_docstore_id=index_to_docstore_id)

def retrieve_relevant_chunks(vector_store, query, k=5):
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join([f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in docs])

def generate_with_google(context, query):
    response = model.generate_content(f"Based on the context provided below, answer the following query. Context: {context}\nQuery: {query}\nAnswer:")
    return response.text

def format_chat_history(history, max_turns=5):
    formatted = ""
    for user_input, bot_reply in history[-max_turns:]:
        formatted += f"User: {user_input}\nBot: {bot_reply}\n"
    return formatted

def run_rag_pipeline_with_history(vector_store, query, chat_history, pdf_names, k=5):
    retrieved_text = retrieve_relevant_chunks(vector_store, query, k=k)
    history_context = format_chat_history(chat_history)
    metadata_context = f"Total PDFs uploaded: {len(pdf_names)}\nList of PDFs: {', '.join(pdf_names)}"
    full_context = f"{history_context}\n{metadata_context}\nContext from documents:\n{retrieved_text}"
    response = generate_with_google(full_context, query)
    chat_history.append((query, response))
    return response

def initialize_rag(load_from_disk=False):
    texts = extract_texts_from_folder(PDF_PATH)
    pdf_names = list(texts.keys())

    if not load_from_disk:
        chunks_with_meta = prepare_chunks_with_metadata(texts)
        vector_store = store_embeddings(chunks_with_meta)
    else:
        vector_store = load_embeddings()

    return vector_store, pdf_names

def is_question_answerable(query, context):
    prompt = f"""Given the following context, can the question be answered with high confidence?

                Context:
                {context}

                Question:
                {query}

                Answer with only 'Yes' or 'No'.
            """
    
    response = model.generate_content(prompt).text.strip().lower()
    return response.startswith("yes")

def dummy_web_search(query):
    # Simulate web results (in real usage, integrate DuckDuckGo, Google API, etc.)
    web_context = f"This is mock web search result for the query: {query}"
    return web_context


def smart_query_handler(vector_store, query, chat_history, pdf_names, k=5):
    context = retrieve_relevant_chunks(vector_store, query, k)
    is_answerable = is_question_answerable(query, context)
    
    if is_answerable:
        return run_rag_pipeline_with_history(vector_store, query, chat_history, pdf_names)
    else:
        web_context = dummy_web_search(query)
        response = generate_with_google(web_context, query)
        chat_history.append((query, response))
        return response


if __name__ == "__main__":
    chat_history = []
    vector_store, pdf_names = initialize_rag(load_from_disk=False)

    query = "How many pdfs are provided?"
    answer = smart_query_handler(vector_store, query, chat_history, pdf_names)
    print(answer)
