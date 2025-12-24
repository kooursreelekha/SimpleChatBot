from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os

 
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")  # Load from .env
)

chat = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content="you are a helpful assistant")
]

while True:
    user_input =input("User input must be given here:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input in ("exit","bye","tata"):
        break
    result=chat.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_history)