from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11435",
    temperature=0,
)

questions = [
    "Почему небо синее?",
    "Как работает паровой двигатель?",
    "Что такое инерционное звено первого порядка?",
]

for question in questions:
    response = llm.invoke(question)

    print(f"\nВопрос: {question}")
    print(f"Ответ: {response.content}")
    print("-" * 60)
