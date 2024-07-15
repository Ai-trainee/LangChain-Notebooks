from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_community.llms import QianfanLLMEndpoint
from langchain.pydantic_v1 import BaseModel, Field
import os

# Set environment variables for authentication
os.environ["QIANFAN_AK"] = "xx"
os.environ["QIANFAN_SK"] = "xx"

# Initialize the Qianfan model endpoint
model = QianfanLLMEndpoint(
    streaming=True,
    model="Yi-34B-Chat",
    endpoint="eb-instant",
)


# Define the output schema using Pydantic
class ExtractedInfo(BaseModel):
    person: str = Field(..., description="Person's name")
    action: str = Field(..., description="Action performed")
    object: str = Field(..., description="Object of the action")
    date: str = Field(..., description="Date of the action")


# Define a prompt for information extraction
prompt = ChatPromptTemplate.from_template(
    "Extract the key information from the following text: {text}"
)


# Function to manually parse the output into the desired structure
def manual_parse_output(text: str) -> ExtractedInfo:
    # Initialize default values
    person, action, object_, date = None, None, None, None

    # Try to parse each line carefully
    lines = text.split("\n")
    for line in lines:
        if "person" in line:
            person = line.split(":")[1].strip() if ":" in line else None
        elif "action" in line:
            action = line.split(":")[1].strip() if ":" in line else None
        elif "object" in line:
            object_ = line.split(":")[1].strip() if ":" in line else None
        elif "date" in line:
            date = line.split(":")[1].strip() if ":" in line else None

    # Create an ExtractedInfo instance, handling missing values
    info = ExtractedInfo(
        person=person or "Unknown",
        action=action or "Unknown",
        object=object_ or "Unknown",
        date=date or "Unknown"
    )
    return info


# Create the chain and run the model
runnable = prompt | model | StrOutputParser()

# Sample text for extraction
sample_text = "张华在2023年5月获得了诺贝尔奖"

# Get the raw output from the model
raw_output = runnable.invoke({"text": sample_text})

# Parse the raw output manually
structured_output = manual_parse_output(raw_output)

# Print the structured output
print(structured_output.json(indent=2))
