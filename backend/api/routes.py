from fastapi import APIRouter,Form,File,UploadFile,HTTPException 
from services.rag_services import process_pdf_and_store,answer_query 
from typing import List


router=APIRouter()



router.post('/upload')
async def upload_pdf(pdf_docs:list[UploadFile]=File(...)):
    try:
        result=process_pdf_and_store(pdf_docs)
        return {'message':result}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    

router.post('/chat')
async def chat_with_docs(user_query:str=Form(...)):
    try:
        response=answer_query(user_query)
        return {'answer':response}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))    