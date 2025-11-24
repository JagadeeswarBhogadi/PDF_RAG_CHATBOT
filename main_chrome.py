import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import hashlib

# -------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------
st.set_page_config(page_title="📘 PDF RAG Chat", layout="wide")
st.title("📄 PDF RAG Chatbot ")

# -------------------------------------------------------
# Initialize Chat History
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------------
# Helper: Hash PDF for unique caching
# -------------------------------------------------------
def get_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# -------------------------------------------------------
# Cache Embedding Model
# -------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    return embed_model

# -------------------------------------------------------
# Cache Vector Store
# -------------------------------------------------------
@st.cache_resource
def build_vectorstore(pdf_bytes, file_hash):
    pdf_path = f"data/{file_hash}.pdf"
    os.makedirs("data", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    

    embed_model = load_embedding_model()

    persist_dir = f"chroma_db/{file_hash}"
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embed_model,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore, len(texts)

# -------------------------------------------------------
# Cache Language Model
# -------------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "stabilityai/stablelm-zephyr-3b"
    offload_dir = "offload_weights"
    os.makedirs(offload_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )
    return qa_pipeline

# -------------------------------------------------------
# PDF Upload Section
# -------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = get_pdf_hash(file_bytes)

    st.info("Building or loading embeddings... ⏳")
    vectorstore, chunks = build_vectorstore(file_bytes, file_hash)
    st.success(f"✅ Vector store ready with {chunks} text chunks.")

    # Load LLM (cached)
    with st.spinner("Loading LLM... This may take 1–2 minutes initially"):
        qa_pipeline = load_llm()
    st.success("✅ LLM loaded and ready!")

    # -------------------------------------------------------
# User Query Section
# -------------------------------------------------------
    query = st.text_input("💬 Ask a question about the uploaded PDF:")

    if query:
       docs = vectorstore.similarity_search(query, k=2)
       context = " ".join([d.page_content for d in docs])

       prompt = f"""You are a helpful assistant. Based on the information provided below, answer the user's question clearly and concisely.

                Context:{context}


                Question:{query} 

                Answer:
                """
       
       #prompt = query
       with st.spinner("🤔 Generating answer..."):
            response = qa_pipeline(prompt)[0]["generated_text"]
            
            # Extract only the answer part
            if "Answer:" in response:
               answer = response.split("Answer:")[1].strip()
            else:
               answer = response.strip()  # fallback if "Answer:" is missing

            # Display the answer
            st.markdown("### 📝 Answer:")
            st.write(answer)


            # Save to chat history
            st.session_state.chat_history.append((query, answer))

   

#-------------------------------------------------------
# Chat History Section (No Context)
# -------------------------------------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧠 Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown("---")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()