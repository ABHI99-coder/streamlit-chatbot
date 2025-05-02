import streamlit as st
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests

#Configure the GOOGLE_API_KEY to use for text embedding
#Use try and catch block to check API expiry
load_dotenv(".env")
GOOGLE_API_KEY = os.getenv("API_Key")
print(GOOGLE_API_KEY)
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def check_api_key():
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Hello, Gemini!"}]}]
        }

        response = requests.post(f"{API_URL}?key={GOOGLE_API_KEY}", json=payload, headers=headers)

        if response.status_code == 401:
            print("API key expired or invalid!")
        elif response.status_code == 403:
            print("API key lacks permissions. Check Google AI Studio settings.")
        elif response.status_code == 404:
            print("API endpoint not found! Verify the API URL.")
        elif response.status_code == 200:
            print("API Key is valid!")
        else:
            print(f"Unexpected Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")

check_api_key()
genai.configure(api_key=GOOGLE_API_KEY)

#Create the persistent storage for chromadb and intialize the collection to store embeddings
chroma_client = chromadb.PersistentClient(path="/ChromaStorage/ChromaDB1")
collection = chroma_client.get_or_create_collection(name="document1")

def generate_embeddings(text_chunks):
    embeddings = []
    try:
        for chunk in text_chunks:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="RETRIEVAL_QUERY"  # Optimize for conversational queries
            )
            embeddings.append(response['embedding'])
    except Exception as e:
        # Handle exceptions (e.g., logging)
        print(f"An error occurred: {e}")
    return embeddings

def query_chromadb(conversation_history, threshold=0.8):
    # Generate embedding for the entire conversation history
    query_embedding = generate_embeddings([conversation_history])[0]
    
    # Perform the query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    
    retrieved_docs = results['metadatas'][0]
    scores = results['distances'][0]
    filtered_docs = [
        doc['content'] for doc, score in zip(retrieved_docs, scores) if score < threshold
    ]

    if filtered_docs:
        # Concatenate the contents of the filtered documents
        concatenated_content = "\n\n".join(filtered_docs)
        print(concatenated_content)
        # Create the prompt for the Gemini API
        prompt = f"User Query: {conversation_history}\n\nIn this query,you may find series of questions but answer the last question only. If the question is straight forward and you can fetch necessary information from the concatenated text then go for it.If you need some context, then please check last to last question and then fetch the necessary info from the concatenated text and likewise follow the pattern if required. If you scanned all questions in the query and you could not find any context then return please try another query. Here is the concatenated text:\n\n{concatenated_content}.\n\n Dont add any extra information. Use the concatenated_content only"

        model = genai.GenerativeModel("gemini-1.5-pro")

        answer = model.generate_content(prompt).parts[0].text

        return answer
    
import streamlit as st

st.title("AI Chatbot")

# Initialize session state for messages if not already present
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display existing chat messages
for message in st.session_state['messages']:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask your query:"):
    # Append user's message to session state
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Prepare conversation history for context
    conversation_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['messages']]
    )

    # Display user's message
    with st.chat_message(name="user"):
        st.markdown(user_input)
    
    # Generate assistant's response
    assistant_response = query_chromadb(conversation_history)
    
    # Append assistant's response to session state
    st.session_state['messages'].append({"role": "assistant", "content": assistant_response})
    
    # Display the assistant's response
    with st.chat_message(name="assistant"):
        st.markdown(assistant_response)


