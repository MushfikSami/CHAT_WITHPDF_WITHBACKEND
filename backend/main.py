from fastapi import FASTAPI,Request 
from api.routes import router 
import time
from prometheus_fastapi_instrumentator import Instrumentator


app=FASTAPI()

@app.middleware('http')
async def add_process_time_header(request:Request,call_next):
    start_time=time.time()
    response= await call_next(request)
    process_time=time.time()-start_time 

    response.headers['X-Process-Time']=str(process_time)
    return response  




Instrumentator().instrument(app).expose(app)

app.include_router(router)