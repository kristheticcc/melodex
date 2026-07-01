# Imports
from config import client, model

def main():
    print("Hello from melodex!")
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "What i the capital of China"}
                    ]
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
