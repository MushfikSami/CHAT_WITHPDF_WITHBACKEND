import streamlit as st 
from utils.api_client import process_pdf, get_answer


st.set_page_config(page_title='Chat with PDF')
st.header('Gemini Docs Chatbot')

user_question=st.text_input('Ask a question from the pdf')

if user_question and st.button('Get Answer'):
    answer=get_answer(user_question)
    st.write(answer['answer']) 


with st.sidebar:
    st.title("Menu")
    pdf_docs=st.file_uploader('Upload multiple docs',accept_multiple_files=True)
    if st.button('Submit and process'):
        result=process_pdf(pdf_docs)
        st.success(result['message'])