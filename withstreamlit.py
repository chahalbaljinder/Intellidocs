import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict
import aiofiles

class DocumentQASystem:
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 250):
        # Load environment variables
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize clients
        self.initialize_clients()
        
        # Configuration
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
    def initialize_clients(self):
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            
            self.gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=self.gemini_api_key,
                model_name="models/text-embedding-004"
            )
            
            self.chroma_client = chromadb.PersistentClient(path="chroma_persistence.storage")
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents_qa_collection",
                embedding_function=self.gemini_ef
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize clients: {str(e)}")

    async def process_uploaded_file(self, uploaded_file):
        """Process a single uploaded file"""
        try:
            content = uploaded_file.getvalue().decode()
            return {"id": uploaded_file.name, "text": content}
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return {"id": uploaded_file.name, "text": ""}

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        words = np.array(text.split())
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            end_idx = min(i + self.chunk_size + self.overlap_size, len(words))
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)
        return chunks

    def process_document(self, document: Dict) -> List[Dict]:
        if not document["text"]:
            return []
        chunks = self.split_text(document["text"])
        return [{"id": f"{document['id']}_{i}", "text": chunk} 
                for i, chunk in enumerate(chunks)]

    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 10) -> List:
        if not texts:
            return []
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.gemini_ef(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                st.error(f"Error generating embeddings for batch {i}: {str(e)}")
                embeddings.extend([np.zeros(768) for _ in batch])
        return embeddings

    async def process_files(self, uploaded_files):
        """Process multiple uploaded files"""
        if not uploaded_files:
            raise ValueError("No files uploaded")

        documents = []
        for file in uploaded_files:
            doc = await self.process_uploaded_file(file)
            if doc["text"]:
                documents.append(doc)

        if not documents:
            raise ValueError("No valid documents to process")

        with ThreadPoolExecutor() as executor:
            chunked_docs = list(executor.map(self.process_document, documents))
        chunked_docs = [item for sublist in chunked_docs for item in sublist]

        if not chunked_docs:
            raise ValueError("No chunks generated from documents")

        texts = [doc["text"] for doc in chunked_docs]
        embeddings = self.batch_generate_embeddings(texts)

        try:
            self.collection.upsert(
                ids=[doc["id"] for doc in chunked_docs],
                documents=[doc["text"] for doc in chunked_docs],
                embeddings=embeddings
            )
            return len(chunked_docs)
        except Exception as e:
            raise RuntimeError(f"Failed to upsert to ChromaDB: {str(e)}")

    async def query(self, question: str, top_k: int = 5):
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k
            )
            
            if not results["documents"]:
                return "No relevant information found in the database."

            relevant_chunks = results["documents"][0]
            context = "\n\n".join(relevant_chunks)
            prompt = (
                "You are an assistant for question-answering tasks. Use the following pieces of "
                "retrieved context to answer the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the answer concise."
                f"\n\nContext:\n{context}\n\nQuestion:\n{question}"
            )
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Error during query: {str(e)}"

async def main():
    st.title("📚 Document QA System")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        try:
            st.session_state.qa_system = DocumentQASystem()
            st.success("✅ System initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            return

    # File upload section
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your text files", 
        type=['txt'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            try:
                with st.spinner("Processing documents..."):
                    num_chunks = await st.session_state.qa_system.process_files(uploaded_files)
                st.success(f"✅ Successfully processed {len(uploaded_files)} documents into {num_chunks} chunks!")
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

    # Query section
    st.header("❓ Ask Questions")
    question = st.text_input("Enter your question:")
    
    if question:
        if st.button("Get Answer"):
            with st.spinner("Finding answer..."):
                answer = await st.session_state.qa_system.query(question)
                st.write("Answer:", answer)

if __name__ == "__main__":
    asyncio.run(main())