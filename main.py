from functions.document_reader import table_converting
from functions.qdrant_setup import Qdrant_setup
from functions.llm_setup import ChatOpenRouter
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MergerRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, SemanticChunker


docstore = InMemoryStore()
llm = ChatOpenRouter(
    model_name="qwen/qwen3-30b-a3b:free"
)

question_docs,table_text = table_converting(llm=llm, docstore=docstore)
db = Qdrant_setup()

db.vectorstore_tables.add_documents(question_docs)
parent_splitter = SemanticChunker(db.embeddings_model)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_retriever = ParentDocumentRetriever(vectorstore=db.vectorstore_text, 
                                           docstore=docstore, child_splitter=child_splitter, 
                                           parent_splitter=parent_splitter)
parent_retriever.add_documents(table_text)

multivector_retriever = MultiVectorRetriever(vectorstore=db.vectorstore_tables, docstore=docstore, id_key="doc_id")
combined_retriever = MergerRetriever(retrievers=[multivector_retriever, parent_retriever])


def format_docs_simple(docs):
    """Превращает список документов в единую строку для контекста."""
    formatted_strings = []
    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get('source', 'неизвестно')
        formatted_string = f"{text} [источник: {source}]"
        formatted_strings.append(formatted_string)
    return "\n\n".join(formatted_strings)

rag_prompt = ChatPromptTemplate.from_template(
    "Ты — полезный ассистент. Отвечай на вопрос только на основе предоставленного контекста. В конце ответа укажи источник(и) в формате [источник: file_name.pdf].\n\nКонтекст:\n{context}\n\nВопрос: {question}"
)


rag_chain = {
    "context": combined_retriever | format_docs_simple, 
    "question": RunnablePassthrough()
} | rag_prompt | llm | StrOutputParser()