import streamlit as st
import os
from dotenv import  load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

##Langsmith Tracking

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A chatbot with LLAMA3.1"


##Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant please respond to the user queries"),
        ("user","Question:{question}"),
    ]
)

##Generate Response

def generate_response(question,llm,temperature,max_tokens):
    llm = ChatOllama(model="llama3.1")
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer    


## Streamlit Integration 

##Title of the app
st.title("Q&AA chatbot with Llama3.1")

##Drop Down to select model
llm = st.sidebar.selectbox("select Opensource Model",["llama3.1:8b", "llama3.1:70b","llama3.1:405b"])

#Adjust response parameter

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens= st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)


##Main Interface for user input

st.write("Ask your question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input, llm,temperature, max_tokens)
    st.write(response)
else:
    st.write("Please ask a question")    