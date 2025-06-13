import streamlit as st
import asyncio
import os
import io
from PIL import Image
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import aiofiles
import json
import base64
from webcrawler import WebCrawler

class EnhancedWebsiteRAGSystem:
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 250):
        """
        Initialize the enhanced RAG system
        
        Args:
            chunk_size: Size of text chunks for processing
            overlap_size: Overlap between chunks to maintain context
        """
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
        
        # Setup storage folders
        self.setup_storage()
        
    def setup_storage(self):
        """Setup storage folders for the system"""
        self.crawled_data_dir = "./crawled_data"
        self.crawled_images_dir = "./crawled_images"
        self.user_images_dir = "./user_images"
        
        if not os.path.exists(self.crawled_data_dir):
            os.makedirs(self.crawled_data_dir)
        
        if not os.path.exists(self.crawled_images_dir):
            os.makedirs(self.crawled_images_dir)
            
        if not os.path.exists(self.user_images_dir):
            os.makedirs(self.user_images_dir)
    
    def initialize_clients(self):
        """Initialize all necessary clients"""
        try:
            # Initialize Gemini
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Initialize embedding function
            self.gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=self.gemini_api_key,
                model_name="models/text-embedding-004"
            )
            
            # Initialize ChromaDB - one collection for text, one for images
            self.chroma_client = chromadb.PersistentClient(path="chroma_persistence.storage")
            
            # Text collection
            self.text_collection = self.chroma_client.get_or_create_collection(
                name="website_text_collection",
                embedding_function=self.gemini_ef
            )
            
            # Image collection
            self.image_collection = self.chroma_client.get_or_create_collection(
                name="website_image_collection",
                embedding_function=self.gemini_ef
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize clients: {str(e)}")
    
    async def crawl_website(self, url: str, max_pages: int = 20, max_depth: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """
        Crawl a website and return the crawled data
        
        Args:
            url: The website URL to crawl
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth of crawling
            
        Returns:
            Tuple of pages data and images data
        """
        crawler = WebCrawler(
            base_url=url,
            max_pages=max_pages,
            max_depth=max_depth,
            download_images=True,
            image_folder=self.crawled_images_dir
        )
        
        # Run the crawler
        pages_data, images_data = await crawler.crawl()
        
        # Save the results
        crawler.save_results(output_folder=self.crawled_data_dir)
        
        return pages_data, images_data
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using numpy for better performance
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        words = np.array(text.split())
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            end_idx = min(i + self.chunk_size + self.overlap_size, len(words))
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)
            
        return chunks
    
    def process_page(self, page: Dict) -> List[Dict]:
        """
        Process a single page into chunks with metadata
        
        Args:
            page: The page data dictionary
            
        Returns:
            List of document dictionaries with text and metadata
        """
        if not page.get("text"):
            return []
            
        url = page.get("url", "")
        chunks = self.split_text(page["text"])
        
        # Create document records with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{url.replace('://', '_').replace('/', '_').replace('.', '_')}_{i}"
            documents.append({
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "has_images": len(page.get("images", [])) > 0
                }
            })
            
        return documents
    
    def process_image(self, image: Dict) -> Dict:
        """
        Process a single image with its metadata
        
        Args:
            image: The image data dictionary
            
        Returns:
            Image document dictionary or None if processing fails
        """
        img_url = image.get("url", "")
        filename = image.get("filename")
        
        if not filename or not os.path.exists(os.path.join(self.crawled_images_dir, filename)):
            return None
            
        # Generate description for the image
        alt_text = image.get("alt_text", "")
        page_url = image.get("page_url", "")
        
        # Create image description
        description = f"Image from {page_url}. "
        if alt_text:
            description += f"Description: {alt_text}. "
            
        # Create image record
        img_id = f"img_{filename.split('.')[0]}"
        return {
            "id": img_id,
            "text": description,  # We'll embed the description
            "metadata": {
                "url": img_url,
                "page_url": page_url,
                "filename": filename,
                "alt_text": alt_text,
                "local_path": os.path.join(self.crawled_images_dir, filename)
            }
        }
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 10) -> List:
        """
        Generate embeddings in batches
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
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
                # Add zero embeddings as fallback
                embeddings.extend([np.zeros(768) for _ in batch])
        return embeddings
    
    async def process_crawled_data(self, pages_data: List[Dict], images_data: List[Dict]):
        """
        Process crawled data and store in ChromaDB
        
        Args:
            pages_data: List of page data dictionaries
            images_data: List of image data dictionaries
            
        Returns:
            Tuple of (number of text chunks, number of images) processed
        """
        # Process pages
        with ThreadPoolExecutor() as executor:
            processed_pages = list(executor.map(self.process_page, pages_data))
        
        # Flatten the list of lists
        text_documents = [item for sublist in processed_pages for item in sublist]
        
        if not text_documents:
            raise ValueError("No text documents generated from crawled pages")
            
        # Process images
        with ThreadPoolExecutor() as executor:
            processed_images = list(executor.map(self.process_image, images_data))
        
        # Filter out None values
        image_documents = [img for img in processed_images if img]
        
        # Store text documents in ChromaDB
        if text_documents:
            try:
                texts = [doc["text"] for doc in text_documents]
                ids = [doc["id"] for doc in text_documents]
                metadatas = [doc["metadata"] for doc in text_documents]
                
                # Generate embeddings
                embeddings = self.batch_generate_embeddings(texts)
                
                # Upsert to ChromaDB
                self.text_collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                st.success(f"Successfully stored {len(text_documents)} text chunks in the database")
            except Exception as e:
                raise RuntimeError(f"Failed to store text documents in ChromaDB: {str(e)}")
        
        # Store image documents in ChromaDB
        if image_documents:
            try:
                texts = [doc["text"] for doc in image_documents]
                ids = [doc["id"] for doc in image_documents]
                metadatas = [doc["metadata"] for doc in image_documents]
                
                # Generate embeddings
                embeddings = self.batch_generate_embeddings(texts)
                
                # Upsert to ChromaDB
                self.image_collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                st.success(f"Successfully stored {len(image_documents)} images in the database")
            except Exception as e:
                raise RuntimeError(f"Failed to store image documents in ChromaDB: {str(e)}")
                
        return len(text_documents), len(image_documents)
    
    async def process_user_image(self, uploaded_image) -> Dict:
        """
        Process a single image uploaded by user to provide additional context for the query
        
        Args:
            uploaded_image: The uploaded image file
            
        Returns:
            Dict with image info
        """
        try:
            # Generate a filename
            filename = f"user_{uploaded_image.name}"
            filepath = os.path.join(self.user_images_dir, filename)
            
            # Save the image
            with open(filepath, "wb") as f:
                f.write(uploaded_image.getbuffer())
                
            # Return image info
            return {
                "path": filepath,
                "description": f"User uploaded image: {uploaded_image.name}",
                "is_user_image": True
            }
        except Exception as e:
            st.error(f"Error processing user image {uploaded_image.name}: {str(e)}")
            return None
    
    async def query(self, question: str, additional_images: List[Dict] = None, top_k_text: int = 5, top_k_images: int = 3, include_images: bool = True):
        """
        Query the system for relevant text and images
        
        Args:
            question: The user's question
            additional_images: List of additional images provided by the user for context
            top_k_text: Number of text chunks to retrieve
            top_k_images: Number of images to retrieve
            include_images: Whether to include images in the response
            
        Returns:
            Tuple of (answer text, URLs, image paths, user images)
        """
        try:
            # Get relevant text chunks
            text_results = self.text_collection.query(
                query_texts=[question],
                n_results=top_k_text
            )
            
            relevant_chunks = []
            urls = set()
            
            if text_results["documents"]:
                relevant_chunks = text_results["documents"][0]  # Get the first list since we only have one query
                
                # Extract unique URLs from the metadata
                for i, metadata in enumerate(text_results["metadatas"][0]):
                    urls.add(metadata["url"])
            
            # Get relevant images if requested
            image_paths = []
            if include_images:
                image_results = self.image_collection.query(
                    query_texts=[question],
                    n_results=top_k_images
                )
                
                if image_results["metadatas"]:
                    for metadata in image_results["metadatas"][0]:
                        if "local_path" in metadata and os.path.exists(metadata["local_path"]):
                            image_paths.append({
                                "path": metadata["local_path"],
                                "url": metadata["url"],
                                "page_url": metadata["page_url"],
                                "alt_text": metadata.get("alt_text", "")
                            })
            
            # Add user-provided additional images if any
            user_images = []
            if additional_images:
                for img in additional_images:
                    if os.path.exists(img["path"]):
                        user_images.append(img)
            
            # Generate response with context
            if not relevant_chunks:
                return "No relevant information found in the database.", list(urls), image_paths, user_images
            
            # Create context from text chunks
            context = "\n\n".join(relevant_chunks)
            
            # Add image descriptions to the context if we have images
            image_context = ""
            if image_paths:
                image_context = "\n\nRelevant images from the website:\n"
                for i, img in enumerate(image_paths):
                    image_context += f"{i+1}. {img['alt_text'] or 'Image'} (from {img['page_url']})\n"
            
            # Add user-provided additional images to context
            user_image_context = ""
            if user_images:
                user_image_context = "\n\nAdditional context images provided by the user:\n"
                for i, img in enumerate(user_images):
                    user_image_context += f"{i+1}. {img.get('description', 'User image')}\n"
            
            # Build the prompt with text and image context
            prompt = (
                "You are an assistant for question-answering tasks about a specific website. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Include relevant URLs in your answer when appropriate. "
                "Keep the answer concise but informative."
                f"\n\nContext:\n{context}"
                f"{image_context}"
                f"{user_image_context}"
                f"\n\nQuestion:\n{question}"
                f"\n\nRelevant URLs: {', '.join(list(urls))}"
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return response.text, list(urls), image_paths, user_images
            
        except Exception as e:
            st.error(f"Error during query: {str(e)}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}", [], [], []
    
    def get_image_base64(self, image_path: str) -> str:
        """
        Convert an image to base64 for display in Streamlit
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            st.error(f"Error reading image {image_path}: {str(e)}")
            return ""

async def main():
    st.set_page_config(
        page_title="Enhanced Website RAG System",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Enhanced Website RAG System")
    st.markdown(
        "Enter a website URL to crawl and create a searchable knowledge base. "
        "Then, ask questions about the website content with optional image context!"
    )
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        try:
            st.session_state.rag_system = EnhancedWebsiteRAGSystem()
            st.success("✅ System initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            return

    # Website crawling section
    st.header("🕸️ Crawl Website")
    with st.form("crawl_form"):
        website_url = st.text_input("Enter Website URL:", placeholder="https://example.com")
        
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.number_input("Max Pages to Crawl:", min_value=1, max_value=100, value=20)
        with col2:
            max_depth = st.number_input("Max Crawl Depth:", min_value=1, max_value=5, value=3)
        
        submit_button = st.form_submit_button("Start Crawling")
    
    if submit_button and website_url:
        try:
            with st.spinner("🕸️ Crawling website... This may take a while depending on the site size."):
                pages_data, images_data = await st.session_state.rag_system.crawl_website(
                    url=website_url,
                    max_pages=max_pages,
                    max_depth=max_depth
                )
                
                st.success(f"✅ Successfully crawled {len(pages_data)} pages and found {len(images_data)} images!")
                
                # Process the crawled data
                with st.spinner("🔍 Processing and embedding crawled content..."):
                    num_text_chunks, num_images = await st.session_state.rag_system.process_crawled_data(
                        pages_data=pages_data,
                        images_data=images_data
                    )
                    
                st.success(f"✅ Successfully processed {num_text_chunks} text chunks and {num_images} images!")
                
        except Exception as e:
            st.error(f"❌ Error crawling website: {str(e)}")

    # Query section
    st.header("❓ Ask Questions About the Website")
    question = st.text_input("Enter your question:")
    include_images = st.checkbox("Include relevant images from the website in the results", value=True)
    
    # Additional image upload for context
    st.subheader("Optional: Upload Additional Images for Context")
    st.markdown("You can upload images to provide additional visual context for your query.")
    additional_uploaded_images = st.file_uploader(
        "Upload images for additional context",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if question:
        if st.button("Get Answer"):
            with st.spinner("🔍 Searching and generating answer..."):
                # Process any uploaded images
                additional_images = []
                if additional_uploaded_images:
                    with st.spinner("Processing your uploaded images..."):
                        for img in additional_uploaded_images:
                            img_data = await st.session_state.rag_system.process_user_image(img)
                            if img_data:
                                additional_images.append(img_data)
                
                # Query the system
                answer, urls, image_paths, user_images = await st.session_state.rag_system.query(
                    question=question,
                    additional_images=additional_images,
                    include_images=include_images
                )
                
                # Display the answer
                st.markdown("### Answer:")
                st.markdown(answer)
                
                # Display relevant URLs
                if urls:
                    st.markdown("### Relevant URLs:")
                    for url in urls:
                        st.markdown(f"- [{url}]({url})")
                
                # Display relevant images from website
                if image_paths and include_images:
                    st.markdown("### Relevant Images from Website:")
                    
                    # Create columns for images
                    num_cols = min(3, len(image_paths))
                    cols = st.columns(num_cols)
                    
                    for i, img_data in enumerate(image_paths):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            img_path = img_data["path"]
                            try:
                                img = Image.open(img_path)
                                st.image(
                                    img, 
                                    caption=img_data.get("alt_text", "Image") or "Image",
                                    use_column_width=True
                                )
                                st.markdown(f"[View Original]({img_data['url']})")
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                
                # Display additional user images
                if user_images:
                    st.markdown("### Your Additional Context Images:")
                    
                    # Create columns for user images
                    num_cols = min(3, len(user_images))
                    cols = st.columns(num_cols)
                    
                    for i, img_data in enumerate(user_images):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            img_path = img_data["path"]
                            try:
                                img = Image.open(img_path)
                                st.image(
                                    img, 
                                    caption=img_data.get("description", "Your image"),
                                    use_column_width=True
                                )
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")

    # Show stats about the database
    st.header("📊 Database Statistics")
    try:
        text_count = len(st.session_state.rag_system.text_collection.get(include=[])["ids"])
        image_count = len(st.session_state.rag_system.image_collection.get(include=[])["ids"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text Chunks", text_count)
        with col2:
            st.metric("Images", image_count)
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
