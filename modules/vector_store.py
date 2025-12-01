import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    def __init__(self, data_dir="data", persist_dir="chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def load_and_index(self):
        """
        Loads text files from the data directory, splits them,
        and creates/updates the vector store.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            return

        # Load documents
        loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            print("No documents found to index.")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create/Update Vector Store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        # self.vector_store.persist() # Chroma 0.4+ persists automatically or uses different method

    def get_retriever(self, k=3):
        """
        Returns a retriever object from the vector store.
        """
        if self.vector_store is None:
            # Try to load existing DB if not already loaded
            if os.path.exists(self.persist_dir):
                 self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            else:
                return None

        return self.vector_store.as_retriever(search_kwargs={"k": k})
