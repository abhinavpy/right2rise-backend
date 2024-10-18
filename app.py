# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import requests
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your frontend

# Replace 'YOUR_PALM_API_KEY' with your actual PaLM API key
# Alternatively, set it as an environment variable
API_KEY = os.environ.get('PALM_API_KEY', 'YOUR_PALM_API_KEY')

# Load embeddings from the pickle file
def load_embeddings(filename='embeddings.pkl'):
    """Load embeddings from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load embeddings when the app starts
print("Loading embeddings...")
all_chunks = load_embeddings()
print(f"Loaded {len(all_chunks)} chunks.")

def get_query_embedding(text):
    """Get embedding vector for the query text using PaLM API."""
    url = f'https://generativelanguage.googleapis.com/v1beta2/models/embedding-gecko-001:embedText?key={API_KEY}'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        'text': text
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        embedding = response.json()['embedding']['value']
        return embedding
    else:
        print(f"Error getting query embedding: {response.status_code}, {response.text}")
        return None

def find_similar_chunks(query_embedding, all_chunks, top_k=5):
    """Find top_k most similar chunks to the query embedding."""
    embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
    query_embedding = np.array(query_embedding)
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10)
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    similar_chunks = [all_chunks[i] for i in top_k_indices]
    return similar_chunks

def generate_answer(context, query):
    """Generate an answer using the context and query with PaLM API."""
    url = f'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key={API_KEY}'
    headers = {
        'Content-Type': 'application/json',
    }
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    data = {
        'prompt': {
            'text': prompt
        },
        'temperature': 0.7,
        'maxOutputTokens': 256
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()['candidates'][0]['output']
        return answer.strip()
    else:
        print(f"Error generating answer: {response.status_code}, {response.text}")
        return "I'm sorry, but I couldn't process your request at this time."

@app.route('/api/ask', methods=['POST'])
def answer_query():
    """Handle incoming queries from the frontend."""
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'answer': 'No query provided.'}), 400

    print(f"Received query: {query}")

    # Get embedding for the query
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        return jsonify({'answer': 'Error generating query embedding.'}), 500

    # Find similar chunks
    similar_chunks = find_similar_chunks(query_embedding, all_chunks)
    context = '\n'.join([chunk['text'] for chunk in similar_chunks])

    # Generate answer
    answer = generate_answer(context, query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    # # Ensure you have set your API_KEY
    # if API_KEY == 'YOUR_PALM_API_KEY':
    #     print("Please set your PaLM API key in the script or as an environment variable 'PALM_API_KEY'.")
    # else:
    #     # Run the app
    app.run(host='0.0.0.0', port=5000)

