# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.

from backend.app.services.document_loader import load_documents

path = "data/Wasserstoff Gen-AI Internship Task.pdf"
docs = load_documents(path)
print(f"Loaded {len(docs)} documents")
