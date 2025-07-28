import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
import xml.etree.ElementTree as ET
import re
import subprocess
import time
from langchain.docstore.document import Document
#from multimodal_router import process_image

#def handle_image_query(image_path, image_type):
#    return process_image(image_path, image_type)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "mplus_topics_2025-06-28.xml")

vector_store = None

def ensure_ollama_model(model_name):
    try:
        ollama.show(model_name)
        print(f"Model '{model_name}' is already installed")
        return True
    except:
        print(f"Installing '{model_name}' model...")
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"Model '{model_name}' installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install {model_name}: {str(e)}")
            return False

def start_ollama_server():
    try:
        ollama.list_models()
        print("Ollama server is already running")
        return True
    except:
        print("Starting Ollama server...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(15)
            return True
        except Exception as e:
            print(f"Failed to start Ollama server: {str(e)}")
            return False

def clean_html(raw_html):
    if raw_html is None:
        return ""
    clean_text = re.sub(r'<[^>]+>', '', raw_html)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

def load_medical_data():
    print(f"Loading medical dataset from {DATASET_PATH}...")
    docs = []
    
    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
        
        tree = ET.parse(DATASET_PATH)
        root = tree.getroot()
        
        for topic in root.findall('health-topic'):
            title_elem = topic.find('title')
            title = clean_html(title_elem.text) if title_elem is not None else "Unknown Topic"
            
            summary_elem = topic.find('full-summary')
            content = clean_html(summary_elem.text) if summary_elem is not None else ""
            
            docs.append(f"TOPIC: {title}\nCONTENT: {content}")
            
            related_drugs = topic.find('related-drugs')
            if related_drugs is not None:
                for drug in related_drugs.findall('drug'):
                    drug_name = clean_html(drug.text) if drug.text else "Unknown Drug"
                    docs.append(f"DRUG: {drug_name}\nRELATED TO: {title}")
    
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise
    
    print(f"Loaded {len(docs)} medical documents")
    return docs

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = [Document(page_content=doc) for doc in docs]
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="mistral")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./med_vector_db_mistral",
        collection_metadata={"hnsw:space": "cosine"}
    )
    vector_store.persist()
    print("Vector store created successfully")
    return vector_store

def initialize_vector_store():
    global vector_store
    
    if not ensure_ollama_model("mistral"):
        raise Exception("Mistral model not available")
    
    if not start_ollama_server():
        raise Exception("Ollama server not running")
    
    new_path = "./med_vector_db_mistral"
    
    if not os.path.exists(new_path):
        print("Building new medical knowledge base for Mistral...")
        try:
            docs = load_medical_data()
            chunks = chunk_documents(docs)
            vector_store = create_vector_store(chunks)
        except Exception as e:
            print(f"Error building knowledge base: {str(e)}")
            raise
    else:
        print("Loading existing medical knowledge base...")
        try:
            embeddings = OllamaEmbeddings(model="mistral")
            vector_store = Chroma(
                persist_directory=new_path,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            raise
    
    return vector_store

# MODIFY THIS FUNCTION:
def medical_qa(query):
    global vector_store
    if vector_store is None:
        vector_store = initialize_vector_store()
    
    results = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    prompt = f"""
    You are a medical assistant. Answer based ONLY on the verified information below.
    Answer in 1-3 lines, recommend consulting a healthcare professional when advise on any medicine is asked.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return response['message']['content'] + "\n\nNote: This is not medical advice. Consult a healthcare professional."

if __name__ == "__main__":
    vector_store = initialize_vector_store()
    
    test_query = "What should I know about HbA1C tests?"
    response = medical_qa(test_query)
    print(f"\nMEDICAL ASSISTANT: {response}")