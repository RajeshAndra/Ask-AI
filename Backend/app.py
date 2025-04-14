from func import *

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

agent = agent()
print(agent.invoke({"input":"Summarize the pdf"}))