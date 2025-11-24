
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------
# STEP 1: Load and split PDF
# -----------------------
pdf_path = "data/documents.pdf"  # Put your multi-page PDF here
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split long documents into manageable text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
print(f"✅ Loaded and split {len(docs)} text chunks from the PDF.")

# -----------------------
# STEP 2: Create embeddings (CPU-friendly)
# -----------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)
print("✅ Vector store built successfully.")

# -----------------------
# STEP 3: Initialize a lightweight local model
# -----------------------
model_name = "google/flan-t5-base"  # Small and CPU-compatible
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------
# STEP 4: Define retrieval + generation
# -----------------------
def generate_answer(query, top_k=5):
    """Retrieve top chunks and generate an answer."""
    # 1. Retrieve similar chunks
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([d.page_content for d in docs])

    # 2. Construct prompt
    prompt = (
        f"Answer the question based on the following context:\n\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # 3. Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------
# STEP 5: Run a sample query
# -----------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = generate_answer(query)
        print(f"\n🤖 Answer: {answer}\n")
