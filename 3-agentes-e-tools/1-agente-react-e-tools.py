from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result."""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mocked web search tool. Returns a hardcoded result."""

    data = {
        "Brasil": "Brasília", 
        "France": "Paris", 
        "Germany": "Berlin", 
        "Italy": "Rome", 
        "Spain": "Madrid", 
        "United States": "Washington D.C"
        }

    for country, capital in data.items():
         if country.lower() in query.lower():
           return f"The capital of {country} is {capital}."

    return "I don't know the capital of that country."

llm = ChatOpenAI(model="gpt-5-mini")
tools = [calculator, web_search_mock]

system_prompt = """
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=system_prompt)

questions = [
     "What is the capital of Iran?",
     "What is the capital of France?",
     "How much is 10 + 10?"
]

for question in questions:

    print("\n==============================")

    response = agent.invoke({
       "messages": [
           {"role": "user", "content": question}
       ]
    })

    for msg in response["messages"]:

        if msg.__class__.__name__ == "HumanMessage":
            print(f"HumanMessage(content='{msg.content}')")

        elif msg.__class__.__name__ == "ToolMessage":
            print(f"ToolMessage(content=\"{msg.content}\")")

    print(response["messages"][-1].content)



