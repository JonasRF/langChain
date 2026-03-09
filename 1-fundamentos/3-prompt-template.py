from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke eith mu name!"
)

text = template.format(name="Jonas")
print(text)