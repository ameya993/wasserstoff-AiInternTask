import streamlit as st
import requests
import base64

API_URL = "http://localhost:8002"

st.set_page_config(page_title="Wasserstoff Document Research Chatbot", layout="wide")
st.title("📄 Document Research & Theme Identification Chatbot")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
- **Step 1:** Upload one or more PDF or image documents.
- **Step 2:** Ask a research question to search across your documents.
- **Step 3:** (Optional) Identify synthesized themes across all documents.
""")
st.sidebar.markdown("---")
st.sidebar.info("Example questions:\n- What is the main objective of these documents?\n- Summarize the key findings.\n- What are the common themes?")

# --- 1. Document Upload ---
st.header("1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDFs or images (multiple allowed)", 
    type=["pdf", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Uploading {file.name}..."):
            files = {"file": (file.name, file.getvalue(), file.type)}
            try:
                resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if resp.ok:
                    st.success(f"Uploaded and indexed: {file.name}")
                else:
                    st.error(f"Failed to upload {file.name}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error during upload: {e}")

# --- 1.b. View & Manage Uploaded Files (Current Session) ---
st.header("1.b. View & Manage Uploaded Files (This Session)")
if uploaded_files:
    for file in uploaded_files:
        file_size_kb = round(len(file.getvalue())/1024, 1)
        st.write(f"**{file.name}** ({file_size_kb} KB)")
        if file.type.startswith("image"):
            st.image(file, width=128)
        elif file.type == "application/pdf":
            b64 = base64.b64encode(file.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{file.name}">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.write("Preview not supported. You can download the file.")
else:
    st.info("No files uploaded yet in this session.")



# --- 2. Query Interface ---
st.header("2. Ask a Research Question")
if "history" not in st.session_state:
    st.session_state["history"] = []
query = st.text_input("Enter your research question:")

if st.button("Get Answers") and query:
    with st.spinner("Retrieving answer..."):
        try:
            resp = requests.post(f"{API_URL}/query", params={"query": query})
            if resp.ok:
                data = resp.json()
                synthesized_answer = data.get("synthesized_answer")
                if synthesized_answer:
                    st.subheader("Synthesized Answer (Llama 3)")
                    st.success(synthesized_answer)
                    st.session_state["history"].append({"q": query, "a": synthesized_answer})
                else:
                    st.info("No relevant answer found.")
            else:
                st.error(f"Error: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# --- 2.b. Session History ---
st.header("2.b. Recent Q&A (Session History)")
if st.session_state["history"]:
    for item in reversed(st.session_state["history"][-5:]):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(f"**A:** {item['a']}")
        st.markdown("---")
else:
    st.info("No questions asked yet in this session.")

# --- 3. Theme Identification and Synthesis ---
st.header("3. Identify Themes Across Documents")
if st.button("Identify Themes") and query:
    with st.spinner("Synthesizing themes..."):
        try:
            resp = requests.post(f"{API_URL}/themes", params={"query": query})
            if resp.ok:
                data = resp.json()
                themes = data.get("themes", [])
                if themes:
                    st.subheader("Synthesized Themes")
                    for idx, theme in enumerate(themes):
                        st.markdown(f"- {theme}")
                else:
                    st.info("No themes identified.")
            else:
                st.error(f"Error: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# --- 4. (Optional) Delete Files from Server ---
st.header("4. Delete Files from Server (Optional)")
try:
    resp = requests.get(f"{API_URL}/files")
    if resp.ok:
        files = resp.json().get("files", [])
        if files:
            delete_file = st.selectbox("Select a file to delete", files)
            if st.button("Delete Selected File"):
                del_resp = requests.delete(f"{API_URL}/delete_file", params={"filename": delete_file})
                if del_resp.ok:
                    st.success(f"Deleted {delete_file} from server.")
                else:
                    st.error(f"Failed to delete {delete_file}: {del_resp.text}")
        else:
            st.info("No documents on server to delete.")
    else:
        st.error("Failed to fetch file list from server.")
except Exception as e:
    st.error(f"Error fetching file list: {e}")
