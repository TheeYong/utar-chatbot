import ollama

# response = ollama.list()
# print(response)

# == Chat example ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
    stream=True,
)
# print(res["message"]["content"]) # Print only response in words

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)