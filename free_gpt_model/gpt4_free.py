from g4f.client import Client
from g4f.Provider.Bing import create_context,create_conversation
import asyncio

client = Client()

prompt = """

what is  the presentation layer:
Source coding (digitization and data compression), and information theory.

"""

System_content = """
A system that can provide step-by-step reasoning and solutions to problems,
utilizing its knowledge and understanding to break down complex queries into simpler parts,
think through each step, and provide a well-reasoned response.

You are an AI assistant designed to perform tasks with the following guidelines:
1. Always provide a step-by-step explanation of your reasoning process.
2. Ensure your responses are as accurate and helpful as possible.
3. If the user's query is unclear, ask for clarification before attempting to answer.
4. Use your knowledge to infer the best course of action if the user's intent is ambiguous."""

User_content = f"""
A user seeking profound insights and guidance on existential questions, expecting clear and detailed responses. 
{prompt}
Note: answer only in english language 

"""

response = client.chat.completions.create(
    
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": System_content
        },
        {
            "role": "user",
            "content": User_content
        }
    ],
    stream=True,
    temperature=0.9,
    max_tokens=100000,
    top_p=1,
    n=1 
)
async def print_response():
  for chunk in response:
     if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content or "", end="")


asyncio.run(print_response()) 




