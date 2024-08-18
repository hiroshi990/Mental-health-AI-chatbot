import streamlit as st
import json
from bot import response



#streamlit page layout
st.set_page_config(
    page_title="How you doin'?",
    page_icon="🧑‍⚕️",
    layout="centered"
    
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
    
    
#streamlit page title

st.title("😄 Mental Therapist Chatbot")

#displaying the chat history for continous chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

#input field for user message
user_query=st.chat_input("Hey,How are you")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role":"user",
                            "content":user_query})
    
    #getting the response
    res=response(user_query)
    st.session_state.chat_history.append({"role":
        "assistant","content":res})
    
    #displaying the message
    with st.chat_message("assistant"):
        st.markdown(res)
    

    
    
    
    
