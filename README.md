# Document QA System 📚  

A robust and interactive Question-Answering system that allows users to upload documents, process them into meaningful chunks, and query for relevant information using Google Gemini AI. The system is built with **Streamlit**, **ChromaDB**, and **Google Generative AI** for a seamless user experience and efficient information retrieval.  

---

## Features ✨  
- **Document Upload**: Upload multiple text files for processing.  
- **Text Chunking**: Splits large documents into manageable chunks for better context retention.  
- **Embedding Generation**: Uses **Google Gemini AI** to generate embeddings for document chunks.  
- **Query System**: Ask questions and receive concise, context-aware answers.  
- **Persistent Storage**: Store and retrieve processed document embeddings using **ChromaDB**.  
- **User-Friendly Interface**: Built with **Streamlit** for easy interaction.  

---

## Skills & Technologies Used 🛠️  
- **NLP**: Text chunking, embeddings, and question answering.  
- **Streamlit**: Interactive UI for document upload and querying.  
- **Google Gemini AI**: Embedding generation and generative responses.  
- **ChromaDB**: Persistent storage for embeddings and document retrieval.  
- **Python Libraries**: `asyncio`, `numpy`, `dotenv`, `concurrent.futures`, and more.  
- **Asynchronous Programming**: Efficient handling of file processing and API calls.  
- **Error Handling**: Robust exception management for a seamless experience.  

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
1. Run the application:  
   ```bash
   streamlit run app.py
   ```  
2. Upload your text documents in the **"Upload Documents"** section.  
3. Process the documents into chunks and store them in the database.  
4. Ask questions in the **"Ask Questions"** section and get concise answers.  

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

---

## Future Enhancements 🌟  
- Support for more file types (e.g., PDF, DOCX).  
- Enhanced embedding models for multilingual support.  
- Real-time updates for document changes.  
- Integration with cloud storage solutions like Google Drive or AWS S3.  

---

## Contributing 🤝  
Contributions are welcome! Feel free to open issues or submit pull requests.  

---

## License 📜  
This project is licensed under the **MIT License**.  

---

## Contact 📧  
For queries or collaboration opportunities, reach out via [LinkedIn](https://www.linkedin.com/in/baljinder-singh).  


