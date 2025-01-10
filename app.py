import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini embedding function
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=gemini_api_key, model_name="models/text-embedding-004")

#Initialize the Chroma client persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistence.storage")
collection_name = "documents_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=gemini_ef)

#Intialize the OpenAI client
genai.configure(api_key=gemini_api_key)

model=genai.GenerativeModel("gemini-1.5-flash")

# question = input("Enter a question: ")

# response = model.generate_content(question)

# print(response.text)

# function to Load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents=[]
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# fuction to split the text into chunks with overlapping
def split_text(text, chunk_size=1000, overlap_size=250):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size + overlap_size]
        chunks.append(chunk)
    return chunks

# Load documents from a directory
directory_path = "./news_articles"
collection = load_documents_from_directory(directory_path)

print(f"loaded {len(collection)} documents from {directory_path}")

#Split documents into chunks
def split_documents_into_chunks(documents):
    print("==== Splitting documents into chunks ====")
    chunked_documents = []
    for document in documents:
        chunks = split_text(document["text"])
        chunked_documents.extend([{"id": document["id"], "text": chunk} for chunk in chunks])
    return chunked_documents

chunked_documents = split_documents_into_chunks(collection)
print(f"Splited {len(collection)} documents into chunks")

# Function to generate embeddings using gemini api
def get_gemini_embedding(text):
    response = gemini_ef(text)
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

# Function to generate embeddings for all chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"])
    print(doc["embedding"])