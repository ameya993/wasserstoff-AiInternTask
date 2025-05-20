# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from pydantic import BaseModel
from app.services.document_loader import load_documents
from app.services.text_splitter import split_documents
from app.services.embedding import get_embedding_model
from app.services.retriever import build_vectorstore
from app.services.reranker import get_reranker_model
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from openai import OpenAI
from typing import List, Optional
import os
import shutil
import traceback
import mimetypes
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime

# --- MongoDB Integration ---
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.environ.get("MONGO_URI")  # Set this in your .env or deployment environment
mongo_client = AsyncIOMotorClient(MONGO_URI)
mongo_db = mongo_client["wasserstoff"]  # Use your DB name

def get_db():
    return mongo_db

router = APIRouter()
VECTOR_STORE_PATH = "data/faiss_index"
INDEX_FILE = os.path.join(VECTOR_STORE_PATH, "index.faiss")
UPLOAD_DIR = "data"

# --- Pydantic models ---
class QueryRequest(BaseModel):
    query: str

class PerDocQueryRequest(BaseModel):
    query: str
    selected_files: Optional[List[str]] = None

class DocAnswer(BaseModel):
    doc_id: str
    document: str
    answer: str
    citation: str

class ThemeOut(BaseModel):
    theme: str
    doc_ids: List[str]
    summary: str

def deduplicate_documents(documents):
    seen = set()
    unique_docs = []
    for doc in documents:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs

def is_image(filename):
    mimetype, _ = mimetypes.guess_type(filename)
    return mimetype and mimetype.startswith('image')

def is_pdf(filename):
    mimetype, _ = mimetypes.guess_type(filename)
    return mimetype == 'application/pdf'

def pdf_has_text(pdf_path):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
        return False
    except Exception as e:
        print(f"[WARN] Could not check PDF text layer: {e}")
        return False

def ocr_image_file(image_path):
    print("[DEBUG] OCR: Running pytesseract on image file...")
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    print("[DEBUG] OCR: Finished extracting text from image.")
    return text

def ocr_pdf_file(pdf_path):
    print("[DEBUG] OCR: Running pytesseract on scanned PDF pages...")
    pages = convert_from_path(pdf_path)
    text = ""
    for page_num, page in enumerate(pages):
        print(f"[DEBUG] OCR: Processing page {page_num+1} of scanned PDF...")
        page_text = pytesseract.image_to_string(page)
        text += f"\n[Page {page_num+1}]\n{page_text}\n"
    print("[DEBUG] OCR: Finished extracting text from scanned PDF.")
    return text

# Utility: Assign next DOC ID (e.g., DOC001)
async def get_next_doc_id(db):
    count = await db.documents.count_documents({})
    return f"DOC{count+1:03d}"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db=Depends(get_db)):
    try:
        print(f"Received upload: {file.filename}")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"Saved file to {save_path}")
    except Exception as e:
        print(f"[ERROR] Saving file failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Assign DOC ID
    doc_id = await get_next_doc_id(db)
    metadata = {
        "doc_id": doc_id,
        "filename": file.filename,
        "type": file.content_type,
        "path": save_path,
        "uploaded_at": datetime.utcnow(),
        # Add more fields as needed (author, doc_type, etc.)
    }
    await db.documents.insert_one(metadata)

    # DELETE existing FAISS index before creating a new one
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"[INFO] Deleting old FAISS index at {VECTOR_STORE_PATH}")
        shutil.rmtree(VECTOR_STORE_PATH)

    # --- OCR or load document ---
    try:
        from langchain_core.documents import Document
        if is_image(file.filename):
            print("[INFO] Detected image file. OCR will be used.")
            text = ocr_image_file(save_path)
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text found in image after OCR. Please upload a clearer image.")
            documents = [Document(page_content=text, metadata={"source": file.filename})]
        elif is_pdf(file.filename):
            if not pdf_has_text(save_path):
                print("[INFO] Detected scanned PDF (no text layer). OCR will be used.")
                text = ocr_pdf_file(save_path)
                if not text.strip():
                    raise HTTPException(status_code=400, detail="No text found in scanned PDF after OCR. Please upload a clearer PDF.")
                documents = [Document(page_content=text, metadata={"source": file.filename})]
            else:
                print("[INFO] Detected digital/text PDF. OCR will NOT be used. Using normal loader.")
                documents = load_documents(save_path)
        else:
            print("[INFO] Unknown file type. OCR will NOT be used. Using normal loader.")
            documents = load_documents(save_path)
        print(f"[INFO] Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"[ERROR] Document loading/OCR failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Document loading/OCR failed: {e}")

    # Split the document
    try:
        print("[INFO] Splitting document...")
        chunks = split_documents(documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks found for embedding. Document may be empty or OCR failed.")
        print(f"[INFO] Split into {len(chunks)} chunks.")
    except Exception as e:
        print(f"[ERROR] Splitting documents failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Split failed: {e}")

    # Embed and build a NEW vectorstore
    try:
        embedding_model = get_embedding_model()
        print("[INFO] Loaded embedding model")
        print("[INFO] Creating new vectorstore...")
        vectorstore = build_vectorstore(chunks, embedding_model)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"[INFO] Saved new vectorstore to {VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"[ERROR] Embedding/vectorstore failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding/vectorstore failed: {e}")

    return {"status": "done", "doc_id": doc_id, "chunks": len(chunks)}

@router.get("/files")
async def list_uploaded_files(
    doc_type: str = Query(None),
    author: str = Query(None),
    date_from: str = Query(None),
    date_to: str = Query(None),
    db=Depends(get_db)
):
    query = {}
    if doc_type:
        query["type"] = doc_type
    if author:
        query["author"] = author
    if date_from or date_to:
        query["uploaded_at"] = {}
        if date_from:
            query["uploaded_at"]["$gte"] = datetime.fromisoformat(date_from)
        if date_to:
            query["uploaded_at"]["$lte"] = datetime.fromisoformat(date_to)
    
    # Return just the filenames as a simple array instead of complex objects
    files = []
    async for doc in db.documents.find(query):
        # Just append the filename string directly
        files.append(doc.get("filename", ""))
    
    return {"files": files}


@router.post("/per_document_query", response_model=List[DocAnswer])
async def per_document_query(request: PerDocQueryRequest, db=Depends(get_db)):
    results = []
    files = request.selected_files or []
    
    if not files:
        # If no files specified, get all filenames
        async for doc in db.documents.find({}):
            files.append(doc["filename"])
    
    print(f"[DEBUG] Processing query for files: {files}")
    
    # Load embedding model once for all documents
    embedding_model = get_embedding_model()
    reranker_model = get_reranker_model()
    
    for filename in files:
        try:
            doc = await db.documents.find_one({"filename": filename})
            if not doc:
                print(f"[WARN] Document not found for filename: {filename}")
                continue
                
            doc_id = doc.get("doc_id", filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            if not os.path.exists(file_path):
                print(f"[WARN] File not found on disk: {file_path}")
                continue
            
            # Load and process the specific document
            print(f"[INFO] Loading document: {filename}")
            documents = []
            
            # Handle different document types
            if is_image(filename):
                text = ocr_image_file(file_path)
                documents = [Document(page_content=text, metadata={"source": filename})]
            elif is_pdf(filename):
                if not pdf_has_text(file_path):
                    text = ocr_pdf_file(file_path)
                    documents = [Document(page_content=text, metadata={"source": filename})]
                else:
                    documents = load_documents(file_path)
            else:
                documents = load_documents(file_path)
            
            # Split the document
            chunks = split_documents(documents)
            
            # Create temporary vectorstore for this document
            temp_vectorstore = build_vectorstore(chunks, embedding_model)
            
            # Set up retrieval pipeline for this document
            retriever = temp_vectorstore.as_retriever(search_kwargs={"k": 5})
            compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            
            # Retrieve relevant chunks
            relevant_chunks = compression_retriever.invoke(request.query)
            unique_chunks = deduplicate_documents(relevant_chunks)
            
            if not unique_chunks:
                results.append(DocAnswer(
                    doc_id=doc_id,
                    document=filename,
                    answer="No relevant information found in this document.",
                    citation="N/A"
                ))
                continue
            
            # Prepare context from chunks
            context = ""
            for chunk in unique_chunks:
                page = chunk.metadata.get("page_label", chunk.metadata.get("page", ""))
                para = chunk.metadata.get("paragraph", "")
                citation = f"Page {page}" if page else "Page ?"
                if para:
                    citation += f", Paragraph {para}"
                context += f"[{citation}] {chunk.page_content}\n\n"
            
            # Generate answer using LLM
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY")
            )
            
            try:
                completion = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context. Be concise and specific. Always cite the page and paragraph numbers."},
                        {"role": "user", "content": f"Context from document '{filename}':\n{context}\n\nQuestion: {request.query}\nAnswer concisely and cite the specific page and paragraph."}
                    ],
                    max_tokens=200,
                    temperature=0.1,
                )
                
                answer = completion.choices[0].message.content.strip()
                
                # Extract citation from the first chunk for simplicity
                # You could implement more sophisticated citation extraction from the answer
                first_chunk = unique_chunks[0]
                page = first_chunk.metadata.get("page_label", first_chunk.metadata.get("page", "?"))
                para = first_chunk.metadata.get("paragraph", "?")
                citation = f"Page {page}, Paragraph {para}"
                
                results.append(DocAnswer(
                    doc_id=doc_id,
                    document=filename,
                    answer=answer,
                    citation=citation
                ))
                
            except Exception as e:
                print(f"[ERROR] LLM processing failed for {filename}: {str(e)}")
                results.append(DocAnswer(
                    doc_id=doc_id,
                    document=filename,
                    answer=f"Error generating answer: {str(e)}",
                    citation="N/A"
                ))
                
        except Exception as e:
            print(f"[ERROR] Error processing document {filename}: {str(e)}")
            traceback.print_exc()
    
    return results



@router.delete("/delete_file")
async def delete_file(filename: str = Query(...), db=Depends(get_db)):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            result = await db.documents.delete_one({"filename": filename})
            if result.deleted_count:
                print(f"[INFO] Successfully deleted file: {filename}")
                return {"status": "deleted", "filename": filename}
            else:
                print(f"[WARN] File {filename} not found in database")
                return {"status": "deleted", "filename": filename, "warning": "File not found in database"}
        else:
            print(f"[WARN] File {filename} not found on disk")
            # Still try to remove from database
            await db.documents.delete_one({"filename": filename})
            return {"error": "File not found on disk", "filename": filename}
    except Exception as e:
        print(f"[ERROR] Error deleting file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.post("/query")
async def query(request: QueryRequest):
    query = request.query
    """
    Query the FAISS vectorstore and return only a synthesized answer using Groq Llama 3.
    """
    try:
        embedding_model = get_embedding_model()
        reranker_model = get_reranker_model()
        if not os.path.exists(INDEX_FILE):
            print("[ERROR] No FAISS index found.")
            raise HTTPException(status_code=404, detail="No FAISS index found. Please upload a document first.")

        print("[INFO] Loading FAISS index...")
        index = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        retriever = index.as_retriever(search_kwargs={"k": 10})
        compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        print("[INFO] Running retrieval and reranking...")
        reranked_documents = compression_retriever.invoke(query)

        # Deduplicate chunks
        unique_docs = deduplicate_documents(reranked_documents)
        print(f"[INFO] Deduplicated to {len(unique_docs)} unique chunks.")

        # Prepare context for Llama 3
        context = ""
        for doc in unique_docs:
            page = doc.metadata.get("page_label", doc.metadata.get("page", ""))
            para = doc.metadata.get("paragraph", "")
            citation = f"[Page {page}" if page else "[Page ?"
            if para:
                citation += f", Paragraph {para}"
            citation += "]"
            context += f"{citation} {doc.page_content}\n"

        print("[DEBUG] Context sent to Llama 3:\n", context)

        # Call Groq Llama 3 for synthesized answer using OpenAI v1.x interface
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        try:
            print("[DEBUG] Calling Groq Llama 3 API...")
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always cite page and paragraph numbers from the context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely and cite page and paragraph numbers where relevant."}
                ],
                max_tokens=300,
                temperature=0.2,
            )
            print("[DEBUG] Raw Groq API response:", completion)
            answer = completion.choices[0].message.content.strip()
            print("[DEBUG] Synthesized answer:", answer)
        except Exception as e:
            print(f"[ERROR] Groq Llama 3 API call failed: {e}")
            traceback.print_exc()
            answer = "Sorry, the Llama 3 API call failed. Please check your Groq API key and try again."

        return {
            "synthesized_answer": answer
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@router.post("/themes")
async def identify_themes(request: QueryRequest):
    query = request.query
    try:
        embedding_model = get_embedding_model()
        reranker_model = get_reranker_model()
        if not os.path.exists(INDEX_FILE):
            print("[ERROR] No FAISS index found.")
            raise HTTPException(status_code=404, detail="No FAISS index found. Please upload a document first.")

        print("[INFO] Loading FAISS index for theme identification...")
        index = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        retriever = index.as_retriever(search_kwargs={"k": 20})  # Get more context for themes
        compressor = CrossEncoderReranker(model=reranker_model, top_n=10)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        print("[INFO] Retrieving chunks for theme synthesis...")
        reranked_documents = compression_retriever.invoke(query)
        unique_docs = deduplicate_documents(reranked_documents)
        print(f"[INFO] Deduplicated to {len(unique_docs)} unique chunks for themes.")

        # Prepare context for Llama 3
        context = ""
        for doc in unique_docs:
            page = doc.metadata.get("page_label", doc.metadata.get("page", ""))
            para = doc.metadata.get("paragraph", "")
            citation = f"[Page {page}" if page else "[Page ?"
            if para:
                citation += f", Paragraph {para}"
            citation += "]"
            context += f"{citation} {doc.page_content}\n"

        print("[DEBUG] Context sent to Llama 3 for themes:\n", context)

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        try:
            print("[DEBUG] Calling Groq Llama 3 API for themes...")
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reads the provided context and identifies common themes across all documents. Respond with a concise bullet list of themes, each theme on a new line."},
                    {"role": "user", "content": f"Context:\n{context}\n\nIdentify the main themes across these documents. Return a bullet list, one theme per line."}
                ],
                max_tokens=300,
                temperature=0.2,
            )
            print("[DEBUG] Raw Groq API response for themes:", completion)
            answer = completion.choices[0].message.content.strip()
            print("[DEBUG] Synthesized themes answer:", answer)
            themes = [line.lstrip("-•* ").strip() for line in answer.splitlines() if line.strip()]
        except Exception as e:
            print(f"[ERROR] Groq Llama 3 API call failed (themes): {e}")
            traceback.print_exc()
            themes = ["Sorry, the Llama 3 API call failed. Please check your Groq API key and try again."]

        return {
            "themes": themes
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Theme synthesis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Theme synthesis failed: {e}")


