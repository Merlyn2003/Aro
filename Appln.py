import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json

# Configuration
LOCAL_LLM_MODEL_PATH = "ggml-model-Q4_K_M.gguf"
QDRANT_URL = "http://localhost:6333"
VECTOR_COLLECTION_NAME = "vector_db"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
LLM_TOP_P = 1
LLM_CONTEXT_LENGTH = 2048
PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
Your name is Aro, and you are a medical chatbot. Introduce yourself in the beginning.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History: {chat_history}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
def initialize_components():
    """Initialize and return components needed for the chatbot."""
    llm = LlamaCpp(
        model_path=LOCAL_LLM_MODEL_PATH,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        top_p=LLM_TOP_P,
        n_ctx=LLM_CONTEXT_LENGTH
    )
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    vector_store = Qdrant(client=qdrant_client, embeddings=embeddings, collection_name=VECTOR_COLLECTION_NAME)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return llm, chain

llm, chain = initialize_components()

def is_greeting(message):
    """Check if the message is a greeting."""
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return any(greeting in message.lower() for greeting in greetings)

def predict(message, history):
    """Generate a response using the conversational chain."""
    # Check if the message is a greeting
    if is_greeting(message):
        greeting_response = "Hello! How can I assist you further?"
        history.append({"role": "assistant", "content": greeting_response})
        return greeting_response, history

    # Convert history to the expected format
    history_langchain_format = [
        {"role": "user", "content": entry["content"]} if entry["role"] == "user" 
        else {"role": "assistant", "content": entry["content"]} 
        for entry in history
    ]

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["chat_history", "question"])
    formatted_prompt = prompt.format(chat_history=json.dumps(history_langchain_format), question=message)

    response = chain({"question": formatted_prompt, "chat_history": history_langchain_format})
    answer = response['answer']

    # Update and return history
    history_langchain_format.append({"role": "user", "content": message})
    history_langchain_format.append({"role": "assistant", "content": answer})
    
    return answer, history_langchain_format

def handle_request():
    """Handle incoming POST requests for chatbot interaction with a timeout."""
    try:
        if request.content_type != 'application/json':
            app.logger.error(f"Unsupported Media Type: {request.content_type}")
            return jsonify({"error": "Content-Type must be application/json"}), 415
        
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        message = data.get('message')
        history = data.get('history', [])

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Set timeout for handling the request
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(predict, message, history)
        try:
            response, updated_history = future.result(timeout=120)  # Set timeout to 120 seconds
            return jsonify({"response": response, "history": updated_history})
        except TimeoutError:
            app.logger.error("Request timed out")
            return jsonify({"error": "Request timed out"}), 504

    except Exception as e:
        app.logger.error(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API Routes
@app.route('/', methods=['POST'])
def chatbot():
    return handle_request()

@app.route('/api', methods=['POST'])
def api_webhook():
    return handle_request()

@app.route('/webhooks/rest/webhook', methods=['GET', 'POST'])
def webhook_route():
    if request.method == 'POST':
        return handle_request()
    return jsonify({"message": "GET request received at /webhooks/rest/webhook"}), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "This is the chatbot API. Please use POST requests to interact."})

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return jsonify({"message": "No favicon available"}), 404

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
