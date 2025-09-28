
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import glob
import os

from endpoints.endpoints import TableQuestions

import uuid
from typing import Optional



def _document_reader(folder_path: str) -> dict:
    """Function for reading documents from user

    Args:
        folder_path (str): path to the folder

    Raises:
        ValueError: error when no files in folder

    Returns:
        dict: two variables with text and table dicts
    """
    files = glob.glob(os.path.join(folder_path,"*.pdf"))
    if not files:
        raise ValueError(f"No files in {folder_path}")
    #Создаем пустые списки для текстовых и табличных значений
    all_docs = []
    all_tables = []

    for file_path in files:
        print(f"   - Parsing file: {os.path.basename(file_path)}")
        elements = partition_pdf(filename=file_path, strategy="fast", languages=["rus", "eng"], infer_table_structure=True)
        file_text_parts = []
        file_table_elements = []
        
        for element in elements:
            metadata = {"source": os.path.basename(file_path)}
            if "Table" in str(type(element)):
                # Таблицы оставляем как отдельные элементы
                file_table_elements.append(Document(page_content=str(element.text), metadata=metadata))
            else:
                # Текстовые части просто собираем в список
                file_text_parts.append(str(element.text))
        
        if file_text_parts:
            full_text = "\n\n".join(file_text_parts)
            all_docs.append(Document(page_content=full_text, metadata={"source": os.path.basename(file_path)}))

        if file_table_elements:
            all_tables.extend(file_table_elements)

    return all_docs, all_tables

def table_converting(folder_path: str,docstore: Optional[object],llm: Optional[object]) -> list:
    """converting tables for future embedding system. Generating uniq id for each table and list of possible questions

    Args:
        llm (Optional[object]): llm object from ChatOpenRouter class
        docstore (Optional[object]): store from Langchain

    Returns:
        list: list with pair of uniq id and all info about tables
    """
    text_docs, table_elements = _document_reader(folder_path)
    structured_llm = llm.with_structured_output(TableQuestions)
    generate_questions_chain = (
        {"table_text":RunnablePassthrough} |
        ChatPromptTemplate.from_template("Сгенерируй гипотетические вопросы по таблице:\n{table_text}") |
        structured_llm
        )
    pydantic_objects = generate_questions_chain.batch([el.page_content for el in table_elements], 
                                                      {"max_concurrency": 5})
    
    question_docs = []
    #Для каждой таблицы генерируем уникальный id и заносим в docstore langchain
    for original_table, question_object in zip(table_elements, pydantic_objects):
        
        current_table_id = str(uuid.uuid4())

        docstore.set(current_table_id, original_table)
        
        for single_question in question_object.questions:
            
            doc = Document(
                page_content=single_question,
                metadata={"doc_id": current_table_id}
            )

            question_docs.append(doc)
    return question_docs, text_docs