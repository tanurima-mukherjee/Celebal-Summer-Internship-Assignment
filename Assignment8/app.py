import os
import tempfile
import numpy as np
from flask import Flask, render_template, request, jsonify
from groq import Groq
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.documents = []
    
    def add_documents(self, docs, embeddings):
        self.documents = docs
        self.embeddings = np.array(embeddings)
    
    def similarity_search(self, query_embedding, k=3):
        if len(self.embeddings) == 0:
            return []
        
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]

class RAGChatbot:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = SimpleVectorStore()
        self.documents_loaded = False
    
    def process_document(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = self.extract_pdf_text(file_path)
        elif ext == '.docx':
            text = self.extract_docx_text(file_path)
        elif ext == '.txt':
            text = self.extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        chunks = self.chunk_text(text)
        embeddings = self.embedding_model.encode(chunks)
        
        self.vector_store.add_documents(chunks, embeddings)
        self.documents_loaded = True
    
    def extract_pdf_text(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def extract_docx_text(self, docx_path):
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def extract_txt_text(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text, chunk_size=500):
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    def query(self, question):
        if not self.documents_loaded:
            return "Please upload a document first."
        
        query_embedding = self.embedding_model.encode([question])[0]
        relevant_docs = self.vector_store.similarity_search(query_embedding, k=3)
        
        context = '\n'.join(relevant_docs)
        
        response = self.groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer based on the context:"
            }],
            model="llama3-8b-8192"
        )
        
        return response.choices[0].message.content

chatbot = RAGChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    temp_path = tempfile.mktemp(suffix=ext)
    try:
        file.save(temp_path)
        chatbot.process_document(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return "File uploaded and processed successfully"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get("question")
    if not query:
        return jsonify({"error": "No question provided"}), 400

    answer = chatbot.query(query)
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)
