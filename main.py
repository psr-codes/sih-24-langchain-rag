from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from rag import query_documents

app = FastAPI()


@app.get("/")
async def hello():
    return "Server is running..."


@app.post("/search")
async def search_query(query: str):
    try:

        result = query_documents(query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
