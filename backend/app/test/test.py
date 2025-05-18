from backend.app.services.document_loader import load_documents

path = "data/Wasserstoff Gen-AI Internship Task.pdf"
docs = load_documents(path)
print(f"Loaded {len(docs)} documents")
