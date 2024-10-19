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

# Replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key
# Alternatively, set it as an environment variable
API_KEY = os.environ.get('API_KEY', 'YOUR_GEMINI_API_KEY')

# Load embeddings from the pickle file
def load_embeddings(filename='embeddings.pkl'):
    """Load embeddings from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load embeddings when the app starts
print("Loading embeddings...")
all_chunks = load_embeddings()
print("Loaded {} chunks.".format(len(all_chunks)))

def get_query_embedding(text):
    """Get embedding vector for the query text using the updated Gemini API."""
    model_name = 'models/text-embedding-004'
    url = f'https://generativelanguage.googleapis.com/v1beta/{model_name}:embedContent?key={API_KEY}'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        'content': {
            'parts': [
                {'text': text}
            ]
        }
        # 'outputDimensionality': 768  # Optional: specify if you want a reduced dimension
    }
    try:
        response = requests.post(url, headers=headers, json=data)
    except Exception as e:
        print(f"Error making embedContent API request: {e}")
        return None

    if response.status_code == 200:
        embedding = response.json()['embedding']['values']
        return embedding
    else:
        print("Error getting query embedding: {}, {}".format(response.status_code, response.text))
        return None

def find_similar_chunks(query_embedding, all_chunks, top_k=5):
    """Find top_k most similar chunks to the query embedding."""
    embeddings = np.array([chunk['embedding'] for chunk in all_chunks], dtype=np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10)
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    similar_chunks = [all_chunks[i] for i in top_k_indices]
    return similar_chunks

def sanitize_text(text):
    """Remove all single and double quotes from the text by replacing them with space."""
    sanitized_text = text.replace('"', ' ').replace("'", ' ')
    return sanitized_text

def generate_answer(context, query):
    """Generate an answer using the context and query with the updated Gemini API."""
    model_name = 'models/gemini-1.5-flash-latest'
    url = f'https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={API_KEY}'
    headers = {
        'Content-Type': 'application/json',
    }
    # Sanitize the context by removing single and double quotes
    sanitized_context = sanitize_text(context)
    prompt_text = f"Context:\n{sanitized_context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    # Construct the JSON payload with only required fields
    data = {
        'contents': [
            {
                'parts': [
                    {'text': prompt_text}
                ]
            }
        ]
        # Optionally, include 'temperature' and 'maxOutputTokens' if needed
        # 'temperature': 0.7,
        # 'maxOutputTokens': 256
    }

    # Debug: Print the data being sent
    print("Sending generateContent request with data:", data)
    
    try:
        response = requests.post(url, headers=headers, json=data)
    except Exception as e:
        print(f"Error making generateContent API request: {e}")
        return "I'm sorry, but I couldn't process your request at this time."
    
    if response.status_code == 200:
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                content_parts = candidate['content']['parts']
                answer_text = ''.join([part['text'] for part in content_parts])
                return answer_text.strip()
            else:
                print("No content in the candidate response.")
                return "I'm sorry, but I couldn't process your request at this time."
        else:
            print("No candidates found in the response.")
            return "I'm sorry, but I couldn't process your request at this time."
    else:
        print("Error generating answer: {}, {}".format(response.status_code, response.text))
        return "I'm sorry, but I couldn't process your request at this time."

@app.route('/api/ask', methods=['POST'])
def answer_query():
    """Handle incoming queries from the frontend."""
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'answer': 'No query provided.'}), 400

    print("Received query: {}".format(query))

    # Get embedding for the query
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        return jsonify({'answer': 'Error generating query embedding.'}), 500

    # Find similar chunks
    similar_chunks = find_similar_chunks(query_embedding, all_chunks)
    context = '\n'.join([chunk['text'] for chunk in similar_chunks])

    # Check if context is empty
    if not context:
        return jsonify({'answer': "I'm sorry, but I couldn't find relevant information to answer your question."}), 404

    # Generate answer
    answer = generate_answer(context, query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    # Ensure you have set your API_KEY
    if API_KEY == 'YOUR_GEMINI_API_KEY':
        print("Please set your Gemini API key in the script or as an environment variable 'API_KEY'.")
    else:
        # Run the app on all interfaces, port 5000
        app.run(host='0.0.0.0', port=5000)
