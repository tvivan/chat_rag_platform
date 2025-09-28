import os
from functions.document_reader import table_converting
from functions.qdrant_setup import QdrantSetup
from functions.llm_setup import ChatOpenRouter
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MergerRetriever, ParentDocumentRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

DOCUMENTS_PATH = "./dataset"
def build_rag_pipeline(db, llm, doc_path):
    """
    Основная функция, которая строит и индексирует весь RAG-пайплайн.
    """
    docstore = InMemoryStore()
    print("1. Loading and processing documents...")
    text_docs, question_docs_for_tables = table_converting(
        folder_path=doc_path, 
        llm=llm, 
        docstore=docstore
    )

    # Индексация данных в Qdrant
    print("2. Indexing data into Qdrant...")
    if question_docs_for_tables:
        db.vectorstore_tables.add_documents(question_docs_for_tables)
        print(f"   - Indexed {len(question_docs_for_tables)} questions for tables.")
    
    if text_docs:
        parent_splitter = SemanticChunker(db.embeddings_model)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        parent_retriever = ParentDocumentRetriever(
            vectorstore=db.vectorstore_text, 
            docstore=docstore, 
            child_splitter=child_splitter, 
            parent_splitter=parent_splitter
        )
        parent_retriever.add_documents(text_docs)
        print(f"   - Indexed {len(text_docs)} text documents.")
    else:
        # Создаем пустой ретривер, если текстовых документов нет
        parent_retriever = None

    # Сборка ретриверов
    print("3. Assembling retrievers...")
    multivector_retriever = MultiVectorRetriever(
        vectorstore=db.vectorstore_tables, 
        docstore=docstore, 
        id_key="doc_id"
    )

    retriever_list = [ret for ret in [multivector_retriever, parent_retriever] if ret is not None]
    combined_retriever = MergerRetriever(retrievers=retriever_list)

    # Создание финальной цепочки (RAG Chain)
    print("4. Building the final RAG chain...")
    def format_docs_simple(docs):
        """Превращает список документов в единую строку для контекста."""
        formatted_strings = [
            f"{doc.page_content} [источник: {doc.metadata.get('source', 'неизвестно')}]"
            for doc in docs
        ]
        return "\n\n".join(formatted_strings)

    rag_prompt = ChatPromptTemplate.from_template(
        "Ты — полезный ассистент. Отвечай на вопрос только на основе предоставленного контекста. В конце ответа укажи источник(и) в формате [источник: file_name.pdf].\n\nКонтекст:\n{context}\n\nВопрос: {question}"
    )

    rag_chain = {
        "context": combined_retriever | format_docs_simple, 
        "question": RunnablePassthrough()
    } | rag_prompt | llm | StrOutputParser()
    
    print("--- RAG Pipeline is ready! ---")
    return rag_chain

if __name__ == "__main__":
    llm = ChatOpenRouter(model_name="qwen/qwen3-30b-a3b:free")

    db = QdrantSetup()
    
    rag_chain = build_rag_pipeline(db=db, llm=llm, doc_path=DOCUMENTS_PATH)

    while True:
        question = input("\nВведите ваш вопрос (или 'exit' для выхода): ")
        if question.lower() == 'exit':
            break
        response = rag_chain.invoke(question)
        print("\nОТВЕТ:")
        print(response)