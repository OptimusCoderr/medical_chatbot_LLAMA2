from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone,FAISS
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
from store_index import embeddings
import os


app = Flask(__name__)

vector_db = FAISS.load_local("Faiss_db",embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt":PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                    'temperature':0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type= "stuff",
    retriever = vector_db.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs = chain_type_kwargs)

@app.route("/")
def index():
    return render_template("chat.html")


if __name__ =="__main__":
    app.run(debug=True)