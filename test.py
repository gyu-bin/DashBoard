from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama  # 올바른 import

llm = ChatOllama(model="mistral:7b")

# role = "<<ROLE DESCRIPTION>>"
messages = [
    SystemMessage(f"정체가 뭐야"),
    HumanMessage("당신을 소개해주세요."),
]

response = llm.invoke(messages)
print(response)
