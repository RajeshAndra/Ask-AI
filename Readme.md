#  AI-Powered Educational Platform with Multi-Agentic RAG

An intelligent educational platform that leverages **multi-agentic Retrieval-Augmented Generation (RAG)** to deliver interactive learning. It offers real-time chatbot support, auto-generates MCQs, and provides feedback to help learners grow from their mistakes.

##  Features

-  **Conversational Chatbot**: Engages users in real-time Q&A to reinforce concepts.
-  **Multi-Agent Reasoning**: Agents handle specific tasks—retrieving, generating, evaluating—to deliver contextual and accurate responses.
-  **MCQ Generation**: Dynamically creates multiple-choice questions based on content and user level.
-  **Feedback Engine**: Analyzes wrong answers and offers detailed, personalized feedback for better understanding.
-  **Personalized Learning**: Adapts questions and responses based on user input and performance.

##  Tech Stack

- Python
- LangChain
- Streamlit  
- Gemini 2.0 Flash LLM Model
- Vector Store (FAISS)
- Multi-agent architecture 

##  How It Works

1. The user asks a question or uploads a document.
2. The retrieval agent fetches relevant content.
3. The chatbot agent answers the query using RAG.
4. The MCQ generator creates quiz questions.
5. Based on the user's answers, the feedback agent provides insights.

##  Run Locally

```bash
git clone https://github.com/RajeshAndra/Ask-AI
cd "Ask-AI"
cd "Backend"
```
Rename the `.env_sample` file to `.env` and replace `YOUR_API_KEY` with your Gemini API Key:
    ```
    GEMINI_API_KEY = A_1hsi***iKs-19w
    ```

```bash
pip install -r requirements.txt
streamlit run app.py
```
