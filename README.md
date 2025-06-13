# Document & Website QA System 📚 🌐 

A robust and interactive Question-Answering system that allows users to:
1. Upload documents, process them into meaningful chunks, and query for relevant information
2. Crawl websites, extract text and images, and query the content for information and relevant links

The system is built with **Streamlit**, **ChromaDB**, and **Google Generative AI** for a seamless user experience and efficient information retrieval.  

---

## Features ✨  
- **Document Upload**: Upload multiple text files for processing.  
- **Text Chunking**: Splits large documents into manageable chunks for better context retention.  
- **Embedding Generation**: Uses **Google Gemini AI** to generate embeddings for document chunks.  
- **Query System**: Ask questions and receive concise, context-aware answers.  
- **Persistent Storage**: Store and retrieve processed document embeddings using **ChromaDB**.  
- **User-Friendly Interface**: Built with **Streamlit** for easy interaction.  
- **Website Crawling**: Crawl websites to extract text and images.
- **Image Processing**: Extract and index images from websites for more comprehensive answers.
- **Multi-modal RAG**: Retrieve both text and images relevant to user queries.
- **Link Retrieval**: Get relevant links from crawled websites in answers.
- **User Image Context** (New!): Upload your own images to provide additional context for queries.
- **Advanced AI Model** (New!): Uses Gemini 2.0 Flash for more accurate and helpful responses.

---

## Skills & Technologies Used 🛠️  
- **NLP**: Text chunking, embeddings, and question answering.  
- **Streamlit**: Interactive UI for document upload and querying.  
- **Google Gemini AI**: Embedding generation and generative responses.  
- **ChromaDB**: Persistent storage for embeddings and document retrieval.  
- **Python Libraries**: `asyncio`, `numpy`, `dotenv`, `concurrent.futures`, and more.  
- **Asynchronous Programming**: Efficient handling of file processing and API calls.  
- **Error Handling**: Robust exception management for a seamless experience.  
- **Web Crawling**: Using BeautifulSoup and Selenium for extracting website content.
- **Image Processing**: Using Pillow for handling and processing images.

---

## Installation 🚀  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/document-qa-system.git
   cd document-qa-system
   ```  
2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Set up environment variables in a `.env` file:  
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```  

---

## Usage 🖥️  
### Using the Launcher (Recommended)
The easiest way to run any of the applications is using the launcher:
```bash
python launcher.py
```
This will present a menu where you can choose which application to run:
1. Document QA System
2. Website RAG System
3. Enhanced Website RAG System (with user image context)
4. Exit

### Document QA System
1. Run the document QA application:  
   ```bash
   streamlit run withstreamlit.py
   ```  
2. Upload your text documents in the **"Upload Documents"** section.  
3. Process the documents into chunks and store them in the database.  
4. Ask questions in the **"Ask Questions"** section and get concise answers.  

### Website RAG System
1. Run the website RAG application:  
   ```bash
   streamlit run rag_app.py
   ```  
2. Enter a website URL in the **"Crawl Website"** section.
3. Set the maximum number of pages and crawl depth, then start crawling.
4. Once crawling is complete, ask questions about the website in the **"Ask Questions About the Website"** section.
5. Get answers with relevant text, links, and images.

### Enhanced Website RAG System (New!)
1. Run the enhanced website RAG application:
   ```bash
   streamlit run enhanced_rag.py
   ```
2. Enter the URL of the website you want to crawl.
3. Set the maximum number of pages and crawl depth, then start crawling.
4. Once crawling is complete, ask questions about the website.
5. You can also upload your own images to provide additional context for your queries!
6. Get answers with relevant text, links, and images from both the crawled website and your uploaded images.

---

## Key Components 🧩  
1. **Document Processing**:  
   - Files are split into chunks with overlap to retain context.  
2. **Embedding Generation**:  
   - Embeddings are generated for each chunk using Google Gemini AI.  
3. **ChromaDB Integration**:  
   - Persistent storage for embedding retrieval and querying.  
4. **Query Answering**:  
   - Uses retrieved document chunks as context for answering user queries.  
5. **Web Crawler**:
   - Extracts text and images from websites for processing.
   - Handles JavaScript-heavy websites using Selenium.
6. **Image Processing**:
   - Downloads and processes images from websites for inclusion in answers.
7. **Multi-modal RAG**:
   - Retrieves both text and images relevant to user queries.

---

## Future Enhancements 🌟  
- Support for more file types (e.g., PDF, DOCX).  
- Enhanced embedding models for multilingual support.  
- Real-time updates for document changes.  
- Integration with cloud storage solutions like Google Drive or AWS S3.  
- Advanced web crawling capabilities with custom filters.
- Video content extraction and analysis.
- User authentication and saved search history.
- Export functionality for crawled data.

---

## Contributing 🤝  
Contributions are welcome! Feel free to open issues or submit pull requests.  

---

## License 📜  
This project is licensed under the **MIT License**.  

---

## Contact 📧  
For queries or collaboration opportunities, reach out via [LinkedIn](https://www.linkedin.com/in/baljinder-singh).


