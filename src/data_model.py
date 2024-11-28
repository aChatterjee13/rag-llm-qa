"""Data Model File"""
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Query Request Class"""
    query: str
    