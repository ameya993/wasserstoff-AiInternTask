from typing import List, Any
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredPDFLoader

def load_documents(file_path: str):
    print(f"[LOADER DEBUG 1] Entered load_documents with: {file_path}")
    try:
        print("[LOADER DEBUG 2] Attempting to import PDFPlumberLoader...")
        from langchain_community.document_loaders import PDFPlumberLoader
        print("[LOADER DEBUG 3] Successfully imported PDFPlumberLoader.")

        print("[LOADER DEBUG 4] Creating PDFPlumberLoader instance...")
        loader = PDFPlumberLoader(file_path)
        print("[LOADER DEBUG 5] PDFPlumberLoader instance created.")

        print("[LOADER DEBUG 6] About to call loader.load() on PDFPlumberLoader...")
        docs = loader.load()
        print("[LOADER DEBUG 7] loader.load() call on PDFPlumberLoader completed.")

        print(f"[LOADER DEBUG 8] Number of documents loaded by PDFPlumberLoader: {len(docs)}")
        print("[LOADER DEBUG 9] Returning documents from load_documents (PDFPlumberLoader).")
        return docs

    except Exception as e:
        print(f"[LOADER DEBUG 10] Exception in PDFPlumberLoader: {e}")
        import traceback; traceback.print_exc()
        print("[LOADER DEBUG 11] PDFPlumberLoader failed. Raising RuntimeError.")
        raise RuntimeError(f"PDFPlumberLoader failed for {file_path}: {e}")

