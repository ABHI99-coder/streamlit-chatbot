__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests


api_key = st.secrets["API_KEY"]


def check_api_key():
    try:
        API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": "Hello"}]}]}

        response = requests.post(f"{API_URL}?key={GOOGLE_API_KEY}", json=payload, headers=headers)
        response.raise_for_status()  

        print("API Key is valid!")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")

check_api_key()
genai.configure(api_key=GOOGLE_API_KEY)

chroma_client = chromadb.PersistentClient(path="ChromaStorage/BoardGames")
collection = chroma_client.get_or_create_collection(name="BoardGames")
#chroma_client = chromadb.PersistentClient(path="./chroma_db3")
#collection = chroma_client.get_or_create_collection(name="document3")

def generate_embeddings(text_chunks):
    embeddings = []
    
    for chunk in text_chunks:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk,
            task_type="RETRIEVAL_QUERY" 
        )
        embeddings.append(response['embedding'])
    return embeddings

def query_chromadb(conversation_history, threshold=0.5):

    query_embedding = generate_embeddings([conversation_history])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6,
    )
    
    retrieved_docs = results['metadatas'][0]
    scores = results['distances'][0]
    filtered_docs = [
        doc['content'] for doc, score in zip(retrieved_docs, scores) 
    ]
    questions = conversation_history.split("\n")
    if filtered_docs:
        concatenated_content = "\n\n".join(filtered_docs)
        #print(concatenated_content)
        prompt = f"""
        You are a smart assistant that extracts answers from a document.
        
        Document Text: "{concatenated_content}"
        
        User's Question: "{conversation_history}"
        
        The Document Text may be happazard.Find and return the best possible answer from the Document Text only. If no answer is found, respond with: 'I could not find the desired output in the PDF. Please try another query or try rephrasing it'
        """
        print(prompt)
        model = genai.GenerativeModel("gemini-2.0-flash")

        answer = model.generate_content(prompt).parts[0].text

        return answer

def refine_question(conversation_history, last_question):
    prompt = f"""
You are an AI assistant helping to reframe a user's question for better understanding.

The user has asked a series of related questions:

"{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(conversation_history.splitlines()) if q.strip())}"

The latest question is: "{last_question}"

**Rules:**
1. If the latest question is already complete and makes sense on its own, return it **exactly as it is**.
2. If the latest question depends on context from previous questions, **rewrite it into a fully self-contained, clear, and precise one-liner**.
3. **Do not use vague words like "it" or "that". Always specify the subject clearly.**
4. Keep the response **brief, without any extra explanation—just return the rewritten question.**

Now, reframe the last question accordingly.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    answer = model.generate_content(prompt).parts[0].text
    print(answer)
    return answer
    
st.title("AI Chatbot 2")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])
        
if user_input := st.chat_input("Ask your query:"):
    
    st.session_state['messages'].append({"role": "user", "content": user_input})
        
    conversation_history = "\n".join(
        [msg["content"] for msg in st.session_state['messages'] if msg["role"] == "user"][-5:]  
    )
    print(conversation_history)

    try:
        refined_question = refine_question(conversation_history, user_input)
    except Exception as e:
        st.error("Something went wrong. Please try again later.")
        assistant_response = "Currently experiencing issues. Please try again later."
        print(f"Error: {e}")
        
    with st.chat_message(name="user"):
        st.markdown(user_input)
        
    try:
        assistant_response = query_chromadb(refined_question)
    
    except Exception as e:
        st.error("Oops! Something went wrong. Please try again later.")
        assistant_response = "Currently experiencing issues. Please try again later."
        print(f"Error: {e}")
    
    st.session_state['messages'].append({"role": "assistant", "content": assistant_response})
    
    with st.chat_message(name="assistant"):
        st.markdown(assistant_response)


