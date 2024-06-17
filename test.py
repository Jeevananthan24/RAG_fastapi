import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List
from fastapi.templating import Jinja2Templates
from fastapi import Request
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")

os.environ['GOOGLE_API_KEY'] = 'AIzaSyBaa-bzhusOF_M2WyjnkB1ha7bpbDDxwkk'

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

conversation = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    global conversation
    if conversation is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please upload PDF files first."})
    response = conversation({'question': question})
    chat_history = response['chat_history']
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

@app.post("/upload", response_class=HTMLResponse)
async def upload_files(request: Request, pdf_files: List[UploadFile] = File(...)):
    global conversation
    if not pdf_files:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Please select PDF files to upload."})

    # Use a cross-platform temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        pdf_paths = []
        for file in pdf_files:
            file_location = os.path.join(tmpdirname, file.filename)
            with open(file_location, "wb") as f:
                shutil.copyfileobj(file.file, f)
            pdf_paths.append(file_location)
        
        raw_text = get_pdf_text(pdf_paths)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        conversation = get_conversational_chain(vector_store)
    
    return templates.TemplateResponse("index.html", {"request": request, "success": "PDF files uploaded successfully."})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
