from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

#import os
#print(os.getenv("OPENAI_API_KEY"))

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
message = model.invoke("Hello world")

print(message.content)