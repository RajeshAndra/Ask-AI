import streamlit as st
from func import *
import time
# Fix the import for StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

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
    if 'agent_thoughts' not in st.session_state:
        st.session_state.agent_thoughts = []
    
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
            
            # Display chat history
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        
                        # Only show thought process for assistant messages
                        if message["role"] == "assistant" and "thoughts" in message and message["thoughts"]:
                            with st.expander("View Agent's Thought Process"):
                                st.markdown(message["thoughts"], unsafe_allow_html=True)
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Display user message
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_query)
                
                if st.session_state.agent is None:
                    create_agent_tools()
                
                # Process the query and show assistant response
                with chat_container:
                    with st.chat_message("assistant"):
                        # Create a placeholder for the real-time thought process display
                        thoughts_placeholder = st.empty()
                        
                        # Create a placeholder for the final answer
                        answer_placeholder = st.empty()
                        
                        # Create the custom callback handler
                        st_callback = CustomAgentCallbackHandler(thoughts_placeholder=thoughts_placeholder)
                        
                        with st.spinner("Thinking..."):
                            # Run the agent with the custom callback handler
                            response = st.session_state.agent.run(user_query, callbacks=[st_callback])
                            
                            # Get the HTML of the thought process
                            thoughts_html = st_callback.get_thoughts_html()
                            
                            # Clear the thought process placeholder before showing final answer
                            thoughts_placeholder.empty()
                            
                            # Display the final response
                            answer_placeholder.write(response)
                            
                            # Add an expander for the thought process
                            with st.expander("View Agent's Thought Process"):
                                st.markdown(thoughts_html, unsafe_allow_html=True)
                            
                            # Add the response and thought process to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response,
                                "thoughts": thoughts_html
                            })
                            
                            # Update memory for the agent
                            st.session_state.memory.chat_memory.add_user_message(user_query)
                            st.session_state.memory.chat_memory.add_ai_message(response)
    
    
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

class CustomAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self, thoughts_placeholder):
        self.thoughts_placeholder = thoughts_placeholder
        self.thoughts = []
        self.tool_inputs = []
        self.tool_outputs = []
        self.current_thought = ""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_thought += "<div style='background-color: #0db38c; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        self.current_thought += "<strong style='color: #1a1a1a;'>🧠 Thinking...</strong><br>"
        self.current_thought += f"<pre style='white-space: pre-wrap; overflow-x: auto; color: #1a1a1a;'>{prompts[0][:200]}...</pre>"
        self.current_thought += "</div>"
        self.thoughts.append(self.current_thought)
        self.thoughts_placeholder.markdown("".join(self.thoughts), unsafe_allow_html=True)
        self.current_thought = ""

    
    def on_llm_end(self, response, **kwargs):
        pass

    def on_llm_new_token(self, token, **kwargs):
        pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.current_thought += "<div style='background-color: #3bbb48; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        self.current_thought += f"<strong style='color: #1a1a1a;'>🔧 Using Tool:</strong> <span style='color: #1a1a1a;'>{serialized['name']}</span><br>"
        self.current_thought += f"<strong style='color: #1a1a1a;'>Input:</strong> <span style='color: #1a1a1a;'>{input_str}</span>"
        self.current_thought += "</div>"
        self.thoughts.append(self.current_thought)
        self.thoughts_placeholder.markdown("".join(self.thoughts), unsafe_allow_html=True)
        self.current_thought = ""
        self.tool_inputs.append(input_str)
    
    def on_tool_end(self, output, **kwargs):
        self.current_thought += "<div style='background-color: #ea7c21; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        self.current_thought += "<strong style='color: #1a1a1a;'>🔍 Tool Output:</strong><br>"
        self.current_thought += f"<pre style='white-space: pre-wrap; overflow-x: auto; color: #1a1a1a;'>{output[:500]}...</pre>"
        self.current_thought += "</div>"
        self.thoughts.append(self.current_thought)
        self.thoughts_placeholder.markdown("".join(self.thoughts), unsafe_allow_html=True)
        self.current_thought = ""
        self.tool_outputs.append(output)


    def on_agent_action(self, action, **kwargs):
        self.current_thought += "<div style='background-color: #ffe3e3; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        self.current_thought += f"<strong style='color: #1a1a1a;'>🤔 Action:</strong> <span style='color: #1a1a1a;'>{action.tool}</span><br>"
        self.current_thought += f"<strong style='color: #1a1a1a;'>Action Input:</strong> <span style='color: #1a1a1a;'>{action.tool_input}</span>"
        self.current_thought += "</div>"
        self.thoughts.append(self.current_thought)
        self.thoughts_placeholder.markdown("".join(self.thoughts), unsafe_allow_html=True)
        self.current_thought = ""

    def on_chain_start(self, serialized, inputs, **kwargs):
        pass

    def on_chain_end(self, outputs, **kwargs):
        pass

    def on_text(self, text, **kwargs):
        self.current_thought += "<div style='background-color: #2566ab; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        self.current_thought += f"<strong>💭 Reasoning:</strong><br><pre style='white-space: pre-wrap;'>{text}</pre>"
        self.current_thought += "</div>"
        self.thoughts.append(self.current_thought)
        self.thoughts_placeholder.markdown("".join(self.thoughts), unsafe_allow_html=True)
        self.current_thought = ""

    def get_thoughts_html(self):
        return "".join(self.thoughts)


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
        description="If the question can be answered then, answer it based on the context extracted from PDFs and previous chat history."
    )
    
    is_answerable_tool = Tool(
        name="is_answerable",
        func=is_answerable_wrapper,
        description="Check whether the query can be answered using context from the PDFs and conversation history. Use this tool first for any query"
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
        description="If the question cannot be answered then, search the web to gather information when the document context does not provide an answer."
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