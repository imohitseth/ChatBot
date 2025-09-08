from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import logging
import json
# Flask- main class we use to create web app
# request- holds all the info about an incoming req
# jsonify- converts python dict to json response for sending data over web

# creating a flask web sever instance
app=Flask(__name__)
CORS(app) # This line enables CORS for all routes
# __name__ => spcl variable that represents name of curr module

# Configure the Gemini API client
# we use an environment variable for the API key
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=gemini_api_key)


#logging to the file (records)
logging.basicConfig(
    filename='chat_logs.json',
    level=logging.INFO,
    format='%(message)s'
)

# Initialize the Generative Model
# We specify the model we want to use. You can explore others as well.
model = genai.GenerativeModel('gemini-1.5-flash')

# knowledge base
def build_knowledge_base():
    """
    Loads text, creates embeddings, and builds a FAISS vector store.
    """
    try:
        loader = TextLoader("knowledge_base.txt")
        docs = loader.load()
        # The embedding model converts text into vectors
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        # Create a vector store from the documents and embeddings
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error building knowledge base: {e}")
        return None

# Build the knowledge base when the application starts
knowledge_base = build_knowledge_base()

# --- NEW: LOAD THE SYSTEM PROMPT ---
def load_system_prompt():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return ""

system_prompt = load_system_prompt()


@app.route('/chat',methods=['POST'])
# decorator for diff url
# /chat => specific path for our chat endpoint
# methods=['POST']: very imp, It specifies that this route should only respond to a POST request. A POST request is used to send data to the server (like a user's message), as opposed to a GET request which is used to retrieve data.
def chat():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # A. RETRIEVAL
        # We'll use a retriever that's aware of the conversation history.
        # This is a key part of making the chatbot coherent.
        retriever = knowledge_base.as_retriever()
        
        # B. GENERATION CHAIN
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.7)

        # The chat prompt is the core of our new system. It includes the system prompt, history, and user input.
        chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\nRelevant context:\n{context}"),
    ("user", "{input}"),
])

        # We'll create a chain that combines the retrieved documents and the chat prompt
        # The AI will decide when to use the context and when to ignore it.
        combine_docs_chain = create_stuff_documents_chain(llm, chat_prompt)
        
        # This chain brings it all together: it retrieves documents and then answers the question.
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Invoke the chain to get a response
        response = retrieval_chain.invoke({"input": user_message})

        #log the conversation
        log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "chatbot_response": chatbot_response
        }
        logging.info(json.dumps(log_entry))

        return jsonify({"response": response["answer"]})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to get a response from the AI."}), 500

    # jsonify fn converts this dict into a json obj