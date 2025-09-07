import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

# Imports for using Ollama
from langchain_ollama import OllamaEmbeddings # Ollama Embeddings
from langchain_ollama import ChatOllama # Ollama LLM Model Holder

# Global, Stateless Components
# These are loaded once and shared across all requests because they are stateless.
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="llama3.2")

# Global State for RAG System
# These are managed globally for simplicity.
rag_chain = None
chat_history = ChatMessageHistory()

def setup_rag_chain(retriever):
    """
    Sets up the global RAG chain with a given retriever.
    This is called after a document has been loaded and a vector store is ready.
    """
    global rag_chain

    # 1. Prompt to rephrase the user's question based on history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Prompt to answer the question using the retrieved context
    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine the chains
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    print("RAG chain created successfully.")

def load_document(file_path: str):
    """
    Loads a PDF file, splits it into chunks, and creates a document-specific
    Chroma vector store. This also sets up the RAG chain.

    Args:
        file_path (str): The path to the PDF file.
    """
    vector_store_path = "./chroma_langchain_db"

    # Check if a vector store already exists from a previous upload
    if os.path.exists(vector_store_path):
        try:
            # Attempt to remove the old vector store
            print("Deleting existing vector store...")
            shutil.rmtree(vector_store_path)
            print("Existing vector store deleted.")
        except PermissionError as e:
            # Handle the case where the folder is in use by another process
            print(f"PermissionError: Could not delete the old vector store. "
                  f"Please ensure no other process is using '{vector_store_path}'.")
            print("The application will proceed, but this may cause unexpected behavior.")
            return # Exit the function if deletion fails

    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return
    
    # Load and split the document
    print(f"Loading document from {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # Create and persist the vector store
    print(f"Creating vector store at {vector_store_path}...")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=vector_store_path)
    print(f"Vector store for document created successfully.")

    # Set up the RAG chain with the new vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    setup_rag_chain(retriever)

def ask_question(query: str):
    """
    Uses the global RAG chain to answer a question and updates the chat history.
    """
    global rag_chain
    global chat_history
    
    if rag_chain is None:
        return "No document has been loaded yet. Please upload a PDF first."

    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history.messages
    })
    
    answer = result["answer"]
    
    # Update chat history
    chat_history.add_user_message(query)
    chat_history.add_ai_message(answer)
    
    return answer

def clear_history():
    """
    Clears the global chat history.
    """
    global chat_history
    chat_history.clear()
    print("Chat history cleared.")
