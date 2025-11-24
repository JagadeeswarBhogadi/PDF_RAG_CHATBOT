# main_7b.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, os

# -------------------------------
# 1️⃣ Load PDF
# -------------------------------
pdf_path = "data/documents.pdf"  # Replace with your PDF path
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents from PDF.")

# -------------------------------
# 2️⃣ Split documents into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)
print(f"✅ Split into {len(texts)} text chunks.")

# -------------------------------
# 3️⃣ Load embedding model on GPU
# -------------------------------
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)
print("✅ Embedding model loaded on GPU.")

# -------------------------------
# 4️⃣ Build FAISS vector store
# -------------------------------
vectorstore = FAISS.from_documents(texts, embed_model)
print("✅ Built FAISS vector store.")

# -------------------------------
# 5️⃣ Load a free 7B instruct model (Hybrid GPU/CPU)
# -------------------------------
model_name = "TheBloke/guanaco-7B-HF"

print(f"🧠 Loading {model_name} on GPU...")

offload_dir = "offload_weights"
os.makedirs(offload_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,             # half precision to save VRAM
    device_map="auto",                     # GPU + CPU hybrid
    offload_folder=offload_dir,            # store overflow weights
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("✅ Model successfully loaded on GPU.")

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True   # ✅ enables sampling for temperature to take effect
)

print("\n---🧠 RAG QA System Ready---")

# -------------------------------
# 6️⃣ Interactive Query Loop
# -------------------------------
while True:
    query = input("\n🔍 Ask a question (or type 'exit' to quit): ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting RAG session.")
        break

    # Retrieve top relevant chunks
    docs = vectorstore.similarity_search(query, k=2)
    context = " ".join([d.page_content for d in docs])

    # Build the full prompt for the LLM
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    print("\n🤖 Generating answer...\n")

    # Generate model response
    response = qa_pipeline(prompt)[0]["generated_text"]

    print("🧩 Answer:\n", response)
