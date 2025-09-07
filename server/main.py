import os
import shutil
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pypdf import PdfReader
from fastapi.responses import FileResponse
from pydantic import BaseModel
from schemas import QuestionRequest

# API Action files
import models, schemas, crud
from database import SessionLocal, engine
from functions import load_document, ask_question, clear_history, setup_rag_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to create tables
def create_tables():
    """
    Creates all database tables. This is done explicitly
    to ensure the tables exist before any queries are run.
    """
    logging.info("Attempting to create database tables...")
    models.Base.metadata.create_all(bind=engine)
    logging.info("Database tables created successfully.")


create_tables()

app = FastAPI()

# Allow CORS for frontend
origins = [
    "http://localhost:3000", # Local frontend Origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Path to upload files
UPLOAD_DIRECTORY = "uploaded_files/"

# Ensure the directories exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)



@app.post("/upload/", response_model=schemas.Document)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Ensure the upload directory exists before trying to write to it
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    # Check if the uploaded file is a PDF
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Save the new file to the local filesystem
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Load the document into the RAG system
    load_document(file_location)

    # Create the document metadata entry in the database
    document = schemas.DocumentCreate(filename=file.filename, file_path=file_location)
    db_document = crud.create_document(db=db, document=document)

    return db_document

@app.post("/answer/{document_id}/", response_model=schemas.AnswerResponse)
async def get_pdf_text(document_id: int, request: QuestionRequest, db: Session = Depends(get_db)):
    # Fetch the document from the database
    document = crud.get_document(db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    # Use the RAG system to get the answer
    answer = ask_question(request.question)
    
    # Getting the full text of pdf
    try:
        reader = PdfReader(document.file_path)
        text = "".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        text = "Could not extract text from PDF."
        logger.error(f"Error extracting text for display: {e}")

    return {"document_id": document_id, "text_from_pdf": text, "answer": answer}

@app.get("/text/{document_id}")
async def get_pdf(document_id: int, db: Session = Depends(get_db)):
    # Fetch the document from the database
    document = crud.get_document(db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = document.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk.")

    return FileResponse(file_path, filename=document.filename, media_type="application/pdf")

@app.delete("/text/{document_id}/")
async def delete_pdf(document_id: int, db: Session = Depends(get_db)):
    # Fetch the document from the database
    document = crud.get_document(db, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    file_path = document.file_path
    # Check if the PDF file exists and delete it
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete the document metadata from the database
    crud.delete_document(db=db, document_id=document_id)
    
    # Clear the chat history
    clear_history()
    
    return {"detail": "Document deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
