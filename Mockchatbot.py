from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

model=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")  # Load from .env
)

chat = ChatHuggingFace(llm=model)

st.header("Mock chatBot")
while True:
    user_input=st.text_input("Come Chat with ME!!!")
    st.write("User:",user_input)
    if user_input in ("Bye","End"):
        break
    else:
        result=chat.invoke(user_input)
        st.write("AI:",result.content)



