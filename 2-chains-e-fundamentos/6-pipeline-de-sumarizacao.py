from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

long_text ="""Jenna is a professional cook and today is her first day in a new restaurant. 
She is very excited, but also nervous because she doesn’t want to get late for her first day. 
She has to be there at 8 o’clock and it is already 7. She is not ready yet and she is unsure how she should go to work. 
She is thinking maybe it is too late to take her car. She might get stuck in traffic and then for sure, she will be late. 
The same thing will happen if she calls a taxi. Instead, she thinks she should take the metro. She gets dressed quickly; 
she grabs some coffee and a sandwich from the coffee shop down the road and gets in the next metro going towards the direction 
of the restaurant. Time is running out and she is getting more and more anxious. 
When the doors open to the metro station, she starts running. Happily, she arrives right on time for her first day."""

spliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

parts = spliter.create_documents([long_text])

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# LCEL map stage: summarize each chunk
map_prompt = PromptTemplate.from_template("Write a concise summary of the following text:\n{text}")
map_chain = map_prompt | llm | StrOutputParser()

prepare_map_inputs = RunnableLambda(lambda docs: [{"text": d.page_content} for d in docs])
map_stage = prepare_map_inputs | map_chain.map()

# LCEL reduce stage: combine summaries into one final summary
reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a single concise summary:\n{text}")
reduce_chain = reduce_prompt | llm | StrOutputParser()

prepare_reduce_input = RunnableLambda(lambda summaries: {"text": "\n".join(summaries)})
pipeline = map_stage | prepare_reduce_input | reduce_chain

result = pipeline.invoke(parts)
print(result)