from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

# Load environment variables from .env file
load_dotenv()

# Check if GROQ_API_KEY is set
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

# Load model
llm = "llama-3.3-70b-versatile"  # Corrected model name
model = GroqModel(
    model_name=llm,  # Updated parameter name to match the library's expected argument
    api_key=groq_api_key
)

# Load system prompt
try:
    with open('prompt.txt', 'r') as file:
        system_prompt = file.read()
except FileNotFoundError:
    raise FileNotFoundError("The file 'prompt.txt' was not found. Please ensure it exists in the current directory.")

# Initialize agent
agent = Agent(
    model=model,
    system_prompt=system_prompt,
    retries=2
)
