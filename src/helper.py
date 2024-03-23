#EXTRACT DATA FROM THE PDF
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings,HuggingFaceEmbeddings



#LOAD THE PDF FROM DIRECTORY
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
     

#CREATE TEXT CHUNKS
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


#DOWNLOAD MODEL FROM THE HUGGING FACE HUB
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
