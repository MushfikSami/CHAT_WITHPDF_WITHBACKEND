from fastapi import FASTAPI 
from api.routes import router 


app=FASTAPI()
app.include_router(router)