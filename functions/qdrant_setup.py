import qdrant_client
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_qdrant import Qdrant
import os


class Qdrant_setup:
    """Class for creating qdrant vectorstore for text and table info
    """
    def __init__(self,embedding_name: str = "ai-sage/Giga-Embeddings-instruct-v1.0",
                 db_path: str = "Qdrant_local_db",
                 embedding_device: str = "cuda"):
        """vectorstore_text for saving text embeddings
           vectorstore_tables for saving table embeddings

        Args:
            embedding_name (str, optional): name of embedding model. Defaults to "ai-sage/Giga-Embeddings-instruct-v1.0".
            db_path (str, optional): path to your qdrant db. Defaults to "Qdrant_local_db".
            embedding_device (str, optional): device for embeddings. Defaults to "cuda".
        """

        self.QDRANT_DB_PATH = db_path
        if not os.path.exists(self.QDRANT_DB_PATH):
            os.makedirs(self.QDRANT_DB_PATH) 
            print(f"Created Qdrant directory at: {self.QDRANT_DB_PATH}")

        self.embeddings_model = HuggingFaceInstructEmbeddings(model_name=embedding_name, 
                                                              model_kwargs={"device": embedding_device}, 
                                                              query_instruction="query:", 
                                                              embed_instruction="passage:")
        
        self.client = qdrant_client.QdrantClient(path=self.db_path)

        self.vectorstore_text = Qdrant(
            client=self.client,
            collection_name="text_chunks",
            embeddings=self.embeddings_model
        )
        
        self.vectorstore_tables = Qdrant(
            client=self.client,
            collection_name="table_chunks",
            embeddings=self.embeddings_model
        )
        