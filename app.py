import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate 
from langchain_classic.chains.question_answering import load_qa_chain
import google.generativeai as genai 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model=HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')


def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        doc=PdfReader(pdf)
        for page in doc.pages:
            text+=page.extract_text()
    return text 


def get_text_chunks(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=1000)
    chunks=splitter.split_text(text)
    return chunks 


def get_vector_store(text_chunks):
    embedding=model
    vector_store=FAISS.from_texts(text_chunks,embedding)
    vector_store.save_local('faiss_store')


def get_conversational_chain():
    prompt_template='''
    Answer the questions from the following context. If the answer is not present in the context,
    say answer is not present in the context. Do not give wrong answer:
    
    Context:{context}

    Question: {question}
'''    
    prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])
    model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    chain= load_qa_chain(model,chain_type='stuff',prompt=prompt) 
    return chain 


def user_input(user_query):
    embedding=model
    new_db=FAISS.load_local('faiss_store',embeddings=embedding,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_query)
    chain=get_conversational_chain()
    response=chain(
        {'input_documents':docs,
         'question':user_query,},
         return_only_outputs=True
    )
    st.write(response['output_text'])





def main():
    st.set_page_config('DOC CHATBOT')
    st.header('GEMINI DOCS CHATBOT')

    user_question=st.text_input('Ask a question from the pdf') 

    if user_question:
        user_input(user_question) 

    with st.sidebar:
        st.title("Menu") 
        pdf_docs=st.file_uploader('Upload multiple docs',accept_multiple_files=True)
        if st.button('Submit and process'):
            texts=get_pdf_text(pdf_docs)
            text_chunks=get_text_chunks(texts)
            get_vector_store(text_chunks)
            st.success("Done")


if __name__ =='__main__':
    main()            