import streamlit as st
from func import *

def main():
    st.set_page_config(page_title="RAG Assistant & MCQ Generator", layout="wide")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'mcqs' not in st.session_state:
        st.session_state.mcqs = []
    if 'current_mcq' not in st.session_state:
        st.session_state.current_mcq = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'submitted_answer' not in st.session_state:
        st.session_state.submitted_answer = False
    
    global embedding_model, chat_model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
    
    if st.session_state.vector_store is None:
        files_exist = os.path.exists(FAISS_FILE) and os.path.exists(DOCSTORE_FILE) and os.path.exists(INDEX_TO_DOCSTORE_ID_FILE)
        if files_exist:
            st.session_state.vector_store = initialize_vectorstore(True)
        
    
    tabs = st.tabs(["Chat Assistant", "MCQ Generator", "PDF Upload"])
    
    
    
    with tabs[2]:
        st.header("Upload PDF Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing PDFs..."):
                
                for file in os.listdir(PDF_PATH):
                    file_path = os.path.join(PDF_PATH, file)
                    if os.path.isfile(file_path) and file.endswith('.pdf'):
                        os.remove(file_path)
                
                file_names = []
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(PDF_PATH, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    file_names.append(uploaded_file.name)
                
                
                st.session_state.vector_store = initialize_vectorstore(False)
                
                
                create_agent_tools()
                
                if len(file_names) == 1:
                    st.success(f"File {file_names[0]} uploaded and indexed successfully!")
                else:
                    st.success(f"{len(file_names)} files uploaded and indexed successfully!")
                    with st.expander("Uploaded Files"):
                        for name in file_names:
                            st.write(f"- {name}")
    
    
    with tabs[0]:
        st.header("PDF Chat Assistant")
        
        
        chat_container = st.container()
        
        
        if st.session_state.vector_store is None:
            st.warning("Please upload PDF documents first in the PDF Upload tab.")
        else:
            
            user_query = st.chat_input("Ask something about your documents...")
            
            
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            
            if user_query:
                
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_query)
                
                
                if st.session_state.agent is None:
                    create_agent_tools()
                
                
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = st.session_state.agent.run(user_query)
                            st.write(response)
                            
                            
                            st.session_state.memory.chat_memory.add_user_message(user_query)
                            st.session_state.memory.chat_memory.add_ai_message(response)
                            
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    
    with tabs[1]:
        st.header("MCQ Generator and Quiz")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Generate MCQs")
            topic = st.text_input("Enter the topic for MCQs:")
            difficulty = st.selectbox("Select difficulty level:", ["Easy", "Medium", "Hard"])
            num_questions = st.slider("Number of questions:", 1, 20, 5)
            
            if st.button("Generate MCQs"):
                if st.session_state.vector_store is None:
                    st.error("Please upload PDF documents first in the PDF Upload tab.")
                else:
                    with st.spinner("Generating questions..."):
                        mcqs = generate_mcqs(topic, st.session_state.vector_store, num_questions, difficulty)
                        if mcqs:
                            st.session_state.mcqs = mcqs
                            st.session_state.current_mcq = 0
                            st.session_state.score = 0
                            st.session_state.submitted_answer = False
                            st.success(f"Generated {len(mcqs)} questions!")
                        else:
                            st.error("Failed to generate questions. Try a different topic or check your documents.")
        
        with col2:
            st.subheader("Quiz")
            
            if not st.session_state.mcqs:
                st.info("Generate MCQs to start the quiz.")
            else:
                
                current = st.session_state.mcqs[st.session_state.current_mcq]
                st.markdown(f"**Question {st.session_state.current_mcq + 1} of {len(st.session_state.mcqs)}**")
                st.markdown(f"**{current['Question']}**")
                
                
                options = {
                    "Option_1": current["Option_1"],
                    "Option_2": current["Option_2"],
                    "Option_3": current["Option_3"],
                    "Option_4": current["Option_4"]
                }
                
                user_answer = st.radio("Select your answer:", 
                                      [f"A. {current['Option_1']}", 
                                       f"B. {current['Option_2']}", 
                                       f"C. {current['Option_3']}", 
                                       f"D. {current['Option_4']}"],
                                      key=f"q_{st.session_state.current_mcq}")
                
                
                option_map = {
                    f"A. {current['Option_1']}": "Option_1",
                    f"B. {current['Option_2']}": "Option_2",
                    f"C. {current['Option_3']}": "Option_3",
                    f"D. {current['Option_4']}": "Option_4"
                }
                user_choice = option_map.get(user_answer)
                
                
                if st.button("Submit Answer") and not st.session_state.submitted_answer:
                    st.session_state.submitted_answer = True
                    is_correct = (user_choice == current["Answer"])
                    
                    if is_correct:
                        st.success("Correct! 🎉")
                        st.session_state.score += 1
                    else:
                        st.error("Incorrect ❌")
                        with st.expander("See explanation"):
                            feedback = mcq_feedback(user_choice, current, st.session_state.vector_store)
                            st.write(feedback)
                
                
                st.progress((st.session_state.current_mcq + 1) / len(st.session_state.mcqs))
                
                
                col1, col2, col3 = st.columns(3)
                
                
                with col1:
                    if st.session_state.current_mcq > 0:
                        if st.button("Previous Question"):
                            st.session_state.current_mcq -= 1
                            st.session_state.submitted_answer = False
                            st.rerun()
                
                
                with col2:
                    if st.session_state.current_mcq < len(st.session_state.mcqs) - 1:
                        if st.session_state.submitted_answer:
                            label = "Next Question"
                        else:
                            label = "Skip to Next"
                            
                        if st.button(label):
                            st.session_state.current_mcq += 1
                            st.session_state.submitted_answer = False
                            st.rerun()
                    elif st.session_state.submitted_answer:
                        st.success(f"Quiz completed! Your score: {st.session_state.score}/{len(st.session_state.mcqs)}")
                        
                
                with col3:
                    if st.button("Reset Quiz"):
                        st.session_state.current_mcq = 0
                        st.session_state.score = 0
                        st.session_state.submitted_answer = False
                        st.rerun()

def create_agent_tools():
    if st.session_state.vector_store is None:
        return
    
    
    def rag_query_wrapper(query):
        return rag_query(query, st.session_state.vector_store, st.session_state.memory)
    
    def is_answerable_wrapper(query):
        return is_answerable(query, st.session_state.vector_store, st.session_state.memory)
    
    def summarize_text_wrapper(query):
        return summarize_text(query, st.session_state.vector_store, st.session_state.memory)
    
    def extract_keywords_wrapper(query):
        return extract_keywords(query, st.session_state.vector_store)
    
    
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
    
    
    st.session_state.agent = initialize_agent(
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
        memory=st.session_state.memory
    )

if __name__ == "__main__":
    main()
