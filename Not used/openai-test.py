import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Get configuration from environment variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")


client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_API_ENDPOINT,
    api_key=AZURE_API_KEY,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=AZURE_DEPLOYMENT_NAME
)

print(response.choices[0].message.content)