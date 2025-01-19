import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict
import asyncio
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
        """Initialize all necessary clients"""
        try:
            # Initialize Gemini
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Initialize embedding function
            self.gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=self.gemini_api_key,
                model_name="models/text-embedding-004"
            )
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path="chroma_persistence.storage")
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents_qa_collection",
                embedding_function=self.gemini_ef
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize clients: {str(e)}")

    async def load_document(self, filepath: str) -> Dict:
        """Load a single document asynchronously"""
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as file:
                content = await file.read()
                return {"id": os.path.basename(filepath), "text": content}
        except Exception as e:
            print(f"Error loading document {filepath}: {str(e)}")
            return {"id": os.path.basename(filepath), "text": ""}

    async def load_documents_from_directory(self, directory_path: str) -> List[Dict]:
        """Load documents asynchronously"""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        tasks = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory_path, filename)
                tasks.append(self.load_document(filepath))
        return await asyncio.gather(*tasks)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using numpy for better performance"""
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
        """Process a single document into chunks with embeddings"""
        if not document["text"]:
            return []
            
        chunks = self.split_text(document["text"])
        return [{"id": f"{document['id']}_{i}", "text": chunk} 
                for i, chunk in enumerate(chunks)]

    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 10) -> List:
        """Generate embeddings in batches"""
        if not texts:
            return []
            
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.gemini_ef(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {str(e)}")
                # Add zero embeddings as fallback
                embeddings.extend([np.zeros(768) for _ in batch])
        return embeddings

    async def process_documents(self, directory_path: str):
        """Main processing pipeline"""
        # Load documents asynchronously
        documents = await self.load_documents_from_directory(directory_path)
        valid_documents = [doc for doc in documents if doc["text"]]
        print(f"Loaded {len(valid_documents)} valid documents")

        if not valid_documents:
            raise ValueError("No valid documents found to process")

        # Process documents into chunks using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            chunked_docs = list(executor.map(self.process_document, valid_documents))
        chunked_docs = [item for sublist in chunked_docs for item in sublist]
        print(f"Split into {len(chunked_docs)} chunks")

        if not chunked_docs:
            raise ValueError("No chunks generated from documents")

        # Generate embeddings in batches
        texts = [doc["text"] for doc in chunked_docs]
        embeddings = self.batch_generate_embeddings(texts)

        # Batch upsert to ChromaDB
        try:
            self.collection.upsert(
                ids=[doc["id"] for doc in chunked_docs],
                documents=[doc["text"] for doc in chunked_docs],
                embeddings=embeddings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upsert to ChromaDB: {str(e)}")

    async def query(self, question: str, top_k: int = 5):
        """Query the system"""
        try:
            # Get relevant chunks - Fixed the API parameter name
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k
            )
            
            if not results["documents"]:
                return "No relevant information found in the database."

            relevant_chunks = results["documents"][0]  # Get the first list since we only have one query

            # Generate response
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
            print(f"Error during query: {str(e)}")
            return "Sorry, I encountered an error while processing your question. Please try again."

# Usage example
async def main():
    try:
        qa_system = DocumentQASystem()
        
        # Process documents
        await qa_system.process_documents("./news_articles")
        
        # Query
        question = "tell me about databricks"
        answer = await qa_system.query(question)
        print(answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())