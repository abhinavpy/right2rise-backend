# process_documents.py

import os
import nltk
from pdfminer.high_level import extract_text
import pickle
import requests
import numpy as np

# Initialize NLTK (run this only once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Replace 'YOUR_PALM_API_KEY' with your actual PaLM API key
# Alternatively, set it as an environment variable 'PALM_API_KEY'
API_KEY = os.environ.get('PALM_API_KEY', 'YOUR_PALM_API_KEY')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def read_text_file(file_path):
    """Read text from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def get_all_documents(directory):
    """Get all documents from the specified directory."""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith('.txt'):
            text = read_text_file(filepath)
            documents.append({'filename': filename, 'text': text})
            print(f"Loaded text file: {filename}")
        elif filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
            documents.append({'filename': filename, 'text': text})
            print(f"Loaded PDF file: {filename}")
        else:
            print(f"Skipped file (unsupported format): {filename}")
    return documents

def split_text(text, max_length=500):
    """Split text into chunks of approximately max_length words."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_embedding(text):
    """Get embedding vector for the given text using PaLM API."""
    url = f'https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "content":{
            "parts":[
                {
                    "text": text
                }
            ]
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        embedding = response.json()['embedding']['value']
        return embedding
    else:
        print(f"Error getting embedding: {response.status_code}, {response.text}")
        return None

def process_documents(directory):
    """Process all documents: extract text, split into chunks, generate embeddings."""
    documents = get_all_documents(directory)
    all_chunks = []
    for doc in documents:
        text_chunks = split_text(doc['text'])
        print(f"Processing {doc['filename']} with {len(text_chunks)} chunks")
        for idx, chunk in enumerate(text_chunks):
            embedding = get_embedding(chunk)
            if embedding:
                all_chunks.append({
                    'document': doc['filename'],
                    'chunk_id': idx,
                    'text': chunk,
                    'embedding': embedding
                })
                print(f"Generated embedding for chunk {idx+1}/{len(text_chunks)} of {doc['filename']}")
            else:
                print(f"Failed to generate embedding for chunk {idx+1} of {doc['filename']}")
    return all_chunks

def save_embeddings(all_chunks, filename='embeddings.pkl'):
    """Save embeddings to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"Embeddings saved to {filename}")

def load_embeddings(filename='embeddings.pkl'):
    """Load embeddings from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # Ensure you have set your API_KEY
    if API_KEY == 'YOUR_PALM_API_KEY':
        print("Please set your PaLM API key in the script or as an environment variable 'PALM_API_KEY'.")
    else:
        all_chunks = process_documents('./documents')
        save_embeddings(all_chunks)
        print("Document processing complete. Embeddings saved to embeddings.pkl.")
