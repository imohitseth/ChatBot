from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import logging
import json

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

logging.basicConfig(
    filename='chat_logs.json',
    level=logging.INFO,
    format='%(message)s'
)

def load_system_prompt():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return ""

system_prompt = load_system_prompt()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Load the knowledge base on each request
        loader = TextLoader("knowledge_base.txt")
        docs = loader.load()

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.7)
        
        # The chat prompt now uses the loaded documents directly
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nRelevant context:\n{context}"),
            ("user", "{input}"),
        ])
        
        # This chain combines the documents and answers the question directly
        combine_docs_chain = create_stuff_documents_chain(llm, chat_prompt)
        response = combine_docs_chain.invoke({"input": user_message, "context": docs})

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "chatbot_response": response
        }
        logging.info(json.dumps(log_entry))

        return jsonify({"response": response})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to get a response from the AI."}), 500
