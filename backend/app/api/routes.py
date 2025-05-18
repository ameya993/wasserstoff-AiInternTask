# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.






from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.document_loader import load_documents
from app.services.text_splitter import split_documents
from app.services.embedding import get_embedding_model
from app.services.retriever import build_vectorstore
from app.services.reranker import get_reranker_model
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from openai import OpenAI

import os
import shutil
import traceback
import mimetypes

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

router = APIRouter()
VECTOR_STORE_PATH = "data/faiss_index"
INDEX_FILE = os.path.join(VECTOR_STORE_PATH, "index.faiss")
UPLOAD_DIR = "data"

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

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Received upload: {file.filename}")
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", file.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"Saved file to {save_path}")
    except Exception as e:
        print(f"[ERROR] Saving file failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

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

    return {"status": "done", "chunks": len(chunks)}

@router.post("/query")
async def query(query: str):
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
async def identify_themes(query: str):
    """
    Synthesizes common themes across all documents using Groq Llama 3.
    """
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


@router.get("/files")
async def list_uploaded_files():
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"files": []}
        files = os.listdir(UPLOAD_DIR)
        files = [f for f in files if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}
    

@router.delete("/delete_file")
async def delete_file(filename: str = Query(...)):
    import os
    UPLOAD_DIR = "data"
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    else:
        return {"error": "File not found"}