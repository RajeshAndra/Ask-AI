from flask import Flask, request, jsonify
from func import *

app = Flask(__name__)

USE_EXISTING_EMBEDDINGS = True

vector_store = initialize_vectorstore(USE_EXISTING_EMBEDDINGS)

def rag_query_wrapper(query):
    return rag_query(query, vector_store, memory)

def is_answerable_wrapper(query):
    return is_answerable(query, vector_store, memory)

def summarize_text_wrapper(query):
    return summarize_text(query, vector_store, memory)

def extract_keywords_wrapper(query):
    return extract_keywords(query, vector_store)

rag_query_tool = Tool(
    name="rag_query",
    func=rag_query_wrapper,
    description="Answer a question based on the context extracted from PDFs and previous chat history."
)

is_answerable_tool = Tool(
    name="is_answerable",
    func=is_answerable_wrapper,
    description="Check whether the query can be answered using context from the PDFs and conversation history."
)

summarize_doc_tool = Tool(
    name="summarize_text",
    func=summarize_text_wrapper,
    description="Summarize the document using previously extracted keywords and conversation history."
)

extract_keywords_tool = Tool(
    name="extract_keywords",
    func=extract_keywords_wrapper,
    description="Use this tool first to extract the top 10 important keywords from the document. Always run this before summarization."
)

web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Search the web to gather information when the document context does not provide an answer."
)

def agent():
    return initialize_agent(
                tools=[
                    extract_keywords_tool,
                    summarize_doc_tool,
                    rag_query_tool,
                    is_answerable_tool,
                    web_search_tool
                ],
                llm=chat_model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory
            )

@app.route("/sendMessage", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("message", "")
    if not query:
        return jsonify({"error": "Query is missing"}), 400
    response = agent.run(query)
    return jsonify({"response": response})

@app.route("/mcq", methods=["POST"])
def handle_mcq():
    data = request.get_json()
    topic = data.get("topic", "")
    difficulty = data.get("difficulty", "Easy")
    count = int(data.get("count", 5))

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    mcqs = generate_mcqs(topic, vector_store, count, difficulty)
    return jsonify({"mcqs": mcqs})

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.get_json()
    user_choice = data.get("user_choice")
    mcq = data.get("mcq")

    if not user_choice or not mcq:
        return jsonify({"error": "user_choice and mcq data are required"}), 400

    feedback = mcq_feedback(user_choice, mcq, vector_store)
    return jsonify({"feedback": feedback})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf = request.files['file']
    if not pdf.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    path = f'uploads/{pdf.filename}'
    pdf.save(path)

    global vector_store
    vector_store = initialize_vectorstore(use_existing_embeddings=False)
    return jsonify({"message": "File uploaded and vector store updated."})


if __name__ == "__main__":
    app.run(debug=True, port = 5353)