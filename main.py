import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END


print("Loading document...")
pdf_path = "Getting_Started.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found. Please place a PDF in the directory.")
    quit()

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ---------------------------------------------------------
# 2. Split the Text & Create Embeddings
# ---------------------------------------------------------
print("Splitting text and building vector database...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------
# 3. AGENTIC RAG SETUP
# ---------------------------------------------------------
@tool(description="Search for information inside the uploaded PDF document. Use this to answer questions about the document.")
def pdf_search(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize the LLM (Ensure you are using a tool-calling capable model like llama3.2)
llm = ChatOllama(model="qwen3:8b", temperature=0)

# Create the agent using LangGraph (Replaces the deprecated AgentExecutor)
agent = create_react_agent(
    model=llm,
    tools=[pdf_search],
    debug=False,
    prompt="You are a helpful assistant. Use the available tools to answer the user's questions accurately if needed. If the information is not in the tools or you are unsure, state clearly that you do not know."
)


def agent_node(state):
    result = agent.invoke({'messages': [('human', state['query'])]})
    state['answer'] = result['messages'][-1].content

    return state


graph = StateGraph(dict)
graph.add_node('agent', agent_node)
graph.set_entry_point('agent')
graph.add_edge('agent', END)
app = graph.compile()


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 4. EXECUTION
    # ---------------------------------------------------------
    while True:
        q = input("query > ")
        if q.lower() == 'quit':
            break
        print(app.invoke({'query':q})['answer'])
