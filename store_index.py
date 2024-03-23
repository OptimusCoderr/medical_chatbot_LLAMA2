from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Chroma,FAISS, Pinecone


import os

def embed_and_merge_index(doc_list, embed_fn, index_store):
    """Function takes in existing vector_store, new doc_list, and embedding function that is initialized on an appropriate model, whether local or online.
    New embedding is merged with the existing index. If no index is given, a new one is created."""
    
    try:
        faiss_db = FAISS.from_documents(doc_list, embed_fn)  
    except Exception as e:
        faiss_db = FAISS.from_texts(doc_list, embed_fn)
    
    if os.path.exists(index_store):
        local_db = FAISS.load_local(index_store, embed_fn)
        local_db.merge_from(faiss_db)
        print("Merge completed")
        local_db.save_local(index_store)
        print("Updated index saved")
        return local_db
    else:
        faiss_db.save_local(folder_path=index_store)
        print("New store created...")
        return faiss_db

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings() 

# Assuming you have defined 'embeddings' somewhere in your code
vector_db = embed_and_merge_index(doc_list=[t.page_content for t in text_chunks],
            embed_fn=embeddings,
            index_store='Faiss_db')