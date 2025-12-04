import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self):
        """
        벡터 저장소 매니저 초기화
        """
        self.data_path = "data"
        self.persist_directory = "chroma_db"
        # 임베딩 모델 설정 (기본값 사용)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def load_and_index(self):
        """
        data 폴더의 파일들을 로드하고 벡터 DB에 저장합니다.
        """
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return

        documents = []
        
        # 1. TXT 파일 로드 (윈도우 한글 오류 해결을 위해 encoding='utf-8' 추가)
        try:
            txt_loader = DirectoryLoader(
                self.data_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"TXT 파일 {len(txt_docs)}개 로드 완료")
        except Exception as e:
            print(f"TXT 로드 중 오류 (무시 가능): {e}")

        # 2. PDF 파일 로드 (PyPDFLoader 사용)
        try:
            pdf_loader = DirectoryLoader(
                self.data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"PDF 파일 {len(pdf_docs)}개 로드 완료")
        except Exception as e:
            print(f"PDF 로드 중 오류 (무시 가능): {e}")

        if not documents:
            print("로드할 문서가 없습니다.")
            return

        # 문서 쪼개기 (Chunking) - 토큰 절약을 위해 사이즈 축소 (1000 -> 500)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        # 벡터 DB 생성 및 저장
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("벡터 DB 인덱싱 완료!")

    def get_retriever(self):
        """
        검색기(Retriever) 반환
        """
        if self.vector_store is None:
            # 이미 저장된 DB가 있는지 확인
            if os.path.exists(self.persist_directory):
                 self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding=self.embeddings
                )
            else:
                return None
        
        # 검색 개수 제한 (k=3) - 토큰 절약
        return self.vector_store.as_retriever(search_kwargs={"k": 3})