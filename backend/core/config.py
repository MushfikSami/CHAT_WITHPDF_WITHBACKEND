import os 

class Settings:
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
    VectorStorePath='../../faiss_store'

settings=Settings()    