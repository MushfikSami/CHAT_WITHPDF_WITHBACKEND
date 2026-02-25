import os 
import requests 


BACKEND_URL=os.getenv('BACKEND_URL','http://localhost:8000')


def process_pdf_and_store(pdf_docs):
    url=f'{BACKEND_URL}/upload'
    files=[('pdf_docs',(pdf.name,pdf,'application/pdf')) for pdf in pdf_docs]
    response=requests.post(url,files=files)
    response.raise_for_status()
    return response.json()


def answer_query(user_query):
    url=f'{BACKEND_URL}/chat'
    response=requests.post(url,data={'user_query':user_query})
    response.raise_for_status()
    return response.json()