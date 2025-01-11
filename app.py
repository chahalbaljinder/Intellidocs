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
    embedding = response[0]
    print("==== Generating embeddings... ====")
    return embedding

# Function to generate embeddings for all chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"])
#print(doc["embedding"])

#upsert documents with embeddings to chromadb
print("==== Inserting chunks into db ====")
for doc in chunked_documents:
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# Function to query documents
def query_documents(question, top_k=5):
    print("==== Querying documents ====")
    #query_embedding = get_gemini_embedding(question)
    results = collection.query(query_text=question, top_k=top_k)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# Function to generate a response from Gemini
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    response = genai.generate_content(
        model="gemini-1.5-flash",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)