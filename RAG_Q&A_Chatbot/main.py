import os
import docx2txt
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT1"] = "RAG_Llama_Chatbot"

# Add a title for your app
st.title("Query Syllabus document using llama3.1")

# Define the directory where Chroma will store the embeddings
persist_dir = "/Users/jayashreekalabhavi/Documents/Open_Source_LLm_Projects/RAG_Q&A_Chatbot/chroma_store"

# Creating llm model
llm = ChatOllama(model="llama3.1")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response possible based on the question.
    <Context>
    {context}
    <Context>
    Question: {input}
    """
)

# Create embeddings and vectors if they don't exist
def create_vectors_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        # Check if we have an existing persisted vector store
        if os.path.exists(persist_dir):
            # Load the persisted vector store
            st.session_state.vectors = Chroma(persist_directory=persist_dir, embedding_function=st.session_state.embeddings)
        else:
            # Process the document and create vectors
            st.session_state.loader = docx2txt.process("Dp_syllabus.docx")
            
            # Split the loaded document text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.create_documents([st.session_state.loader])

            # Initialize Chroma with persist_directory to save the vector embeddings
            st.session_state.vectors = Chroma.from_documents(
                final_documents, 
                st.session_state.embeddings, 
                persist_directory=persist_dir
            )
            
            # Persist the vectors (save them to disk)
            st.session_state.vectors.persist()

# Input for user query
user_prompt = st.text_input("Enter your query from the document")

# Button to create document embeddings
#if st.button("Initialize Document Embeddings"):
create_vectors_embeddings()
    #st.write("Vector database is ready!")

# Process the user query and provide an answer
if user_prompt:
    # # Ensure the vector database is initialized
    # if "vectors" not in st.session_state:
    #     st.write("Please initialize the document embeddings first.")
    # else:
        # Set up document retrieval and chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        # Time the response
        start_time = time.process_time()
        response = retriever_chain.invoke({"input": user_prompt})
        response_time = time.process_time() - start_time
        st.write(f"Response time: {response_time:.2f} seconds")

        # Display the answer
        st.write(f"Assistant: {response['answer']}")

        # Streamlit expander for document similarity search
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("------------------------")
