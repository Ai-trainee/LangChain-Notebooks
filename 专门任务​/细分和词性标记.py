from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
"""For basic init and call"""
import os

from langchain_community.llms import QianfanLLMEndpoint

# Set environment variables for authentication
os.environ["QIANFAN_AK"] = "xx"
os.environ["QIANFAN_SK"] = "xx"



model = QianfanLLMEndpoint(
    streaming=True,
    model="Yi-34B-Chat",
    endpoint="eb-instant",
)
# Define a prompt for segmentation and POS tagging
prompt = ChatPromptTemplate.from_template(
    "Segment the following text and provide POS tags for each word: {text}"
)

output_parser = StrOutputParser()

# Create the chain
chain = prompt | model | output_parser

# Run the chain
result = chain.invoke({"text": "我爱自然语言处理"})
print(result)
