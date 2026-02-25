import os 
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classics.chains.question_answering import load_qa_chain
import google.generativeai as genai
from core.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def process_pdf_and_store(pdf_docs):
    texts=''
    for pdf in pdf_docs:
        doc=PdfReader(pdf.file)
        for page in doc.pages:
            texts+=page.extract_text()


    splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=1000)
    chunks=splitter.split_text(texts)
    vector_store=FAISS.from_texts(chunks,embedding)
    vector_store.save_local(settings.FAISS_STORE_PATH)
    return "PDF processed and stored successfully" 



def answer_query(user_query):
    new_db=FAISS.load_local(settings.FAISS_STORE_PATH,embeddings=embedding,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_query)

    prompt_template='''
    Answer the questions from the following context. If the answer is not present in the context,
    say answer is not present in the context. Do not give wrong answer:

    CONTEXT:{context}

    QUESTION:{question}
'''

    prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])
    model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    chain=load_qa_chain(model,chain_type='stuff',prompt=prompt)

    response=chain(input_documents=docs,question=user_query)
    return response['output_text']