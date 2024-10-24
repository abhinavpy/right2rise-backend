# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import requests
import pickle
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import jwt
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Enable CORS to allow requests from your frontend

# Configuration
API_KEY = os.environ.get('API_KEY', 'YOUR_GEMINI_API_KEY')
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID')  # Add your Client ID
JWT_SECRET = os.environ.get('JWT_SECRET', 'your_jwt_secret_key')  # Replace with a secure key
JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_SECONDS = 3600  # Token valid for 1 hour

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
        print("Error getting embedding: {}, {}".format(response.status_code, response.text))
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

def create_jwt_token(user_info):
    """Create a JWT token."""
    payload = {
        'user_id': user_info['sub'],  # Google's unique user ID
        'email': user_info.get('email'),
        'name': user_info.get('name'),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode('utf-8')

    return token

def verify_jwt_token(token):
    """Verify a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("JWT token has expired.")
        return None
    except jwt.InvalidTokenError:
        print("Invalid JWT token.")
        return None

@app.route('/api/auth/google', methods=['POST'])
def auth_google():
    """Authenticate user using Google ID token."""
    data = request.get_json()
    token = data.get('token', '')

    if not token:
        return jsonify({'success': False, 'message': 'No token provided.'}), 400

    try:
        # Specify the CLIENT_ID of the app that accesses the backend
        idinfo = id_token.verify_oauth2_token(token, grequests.Request(), GOOGLE_CLIENT_ID)

        # ID token is valid. Get the user's Google Account ID from the decoded token.
        userid = idinfo['sub']
        email = idinfo.get('email')
        name = idinfo.get('name')

        # Create JWT token for your application
        jwt_token = create_jwt_token(idinfo)

        return jsonify({'success': True, 'token': jwt_token}), 200

    except ValueError as e:
        # Invalid token
        print(f"Invalid token: {e}")
        return jsonify({'success': False, 'message': 'Invalid token. {e}'}), 400


@app.route('/api/ask', methods=['POST'])
def answer_query():
    """Handle incoming queries from the frontend."""
    # Verify JWT token from Authorization header
    auth_header = request.headers.get('Authorization', '')
    if not auth_header:
        logging.warning("Authorization header missing.")
        return jsonify({'success': False, 'answer': 'Authorization header missing.'}), 401

    try:
        # Expecting header in format "Bearer <token>"
        token = auth_header.split()[1]
    except IndexError:
        logging.warning("Invalid authorization header format.")
        return jsonify({'success': False, 'answer': 'Invalid authorization header format.'}), 401

    payload = verify_jwt_token(token)
    if not payload:
        logging.warning("Invalid or expired token.")
        return jsonify({'success': False, 'answer': 'Invalid or expired token.'}), 401

    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        logging.warning("No query provided by user {}.".format(payload.get('email', 'Unknown')))
        return jsonify({'success': False, 'answer': 'No query provided.'}), 400

    print("Received query from user {}: {}".format(payload['email'], query))
    logging.info("Received query from user {}: {}".format(payload.get('email', 'Unknown'), query))

    # Get embedding for the query
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        logging.error("Error generating query embedding for user {}.".format(payload.get('email', 'Unknown')))
        return jsonify({'success': False, 'answer': 'Error generating query embedding.'}), 500

    # Find similar chunks
    similar_chunks = find_similar_chunks(query_embedding, all_chunks)
    context = '\n'.join([chunk['text'] for chunk in similar_chunks])

    # Check if context is empty
    if not context:
        logging.info("No relevant information found for user {}'s query.".format(payload.get('email', 'Unknown')))
        return jsonify({'success': False, 'answer': "I'm sorry, but I couldn't find relevant information to answer your question."}), 404

    # Generate answer
    answer = generate_answer(context, query)
    return jsonify({'success': True, 'answer': answer})

if __name__ == '__main__':
    # Ensure you have set your API_KEY, GOOGLE_CLIENT_ID, and JWT_SECRET
    if API_KEY == 'YOUR_GEMINI_API_KEY' or GOOGLE_CLIENT_ID == 'YOUR_GOOGLE_CLIENT_ID' or JWT_SECRET == 'your_jwt_secret_key':
        print("Please set your Gemini API key, Google Client ID, and JWT secret key in environment variables.")
    else:
        # Run the app on all interfaces, port 5000
        app.run(host='0.0.0.0', port=5000)

