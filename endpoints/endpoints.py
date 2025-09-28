from pydantic import BaseModel,Field
from typing import List

class TableQuestions(BaseModel):
    """Pydantic class for structured output questions on tables for llm

    Args:
        BaseModel (_type_)
    """
    questions: List[str] = Field(description="Список из 3-5 гипотетических вопросов по таблице. Которые могут быть заданы по ней.")
    