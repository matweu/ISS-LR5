import json

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11435",
    temperature=0,
    format="json",
)

system = SystemMessage(
    content=(
        "Ты отвечаешь на энциклопедические вопросы. "
        "Всегда возвращай только JSON с двумя строковыми полями: "
        "short_answer и full_answer. "
        "short_answer — краткий ответ в несколько слов. "
        "full_answer — полный ответ в несколько предложений. "
        "Не добавляй текст вне JSON."
    )
)

examples = [
    (
        "Какой газ необходим человеку для дыхания?",
        {
            "short_answer": "Кислород",
            "full_answer": (
                "Человеку необходим кислород. "
                "Он используется клетками для получения энергии "
                "в процессе клеточного дыхания."
            ),
        },
    ),
    (
        "Что такое гравитация?",
        {
            "short_answer": "Гравитационное взаимодействие",
            "full_answer": (
                "Гравитация — это взаимодействие между телами, "
                "обладающими массой или энергией. "
                "Она определяет движение планет и падение тел."
            ),
        },
    ),
]

messages = [system]

for question, answer in examples:
    messages.append(HumanMessage(content=question))
    messages.append(
        AIMessage(
            content=json.dumps(answer, ensure_ascii=False)
        )
    )

questions = [
    "Почему небо синее?",
    "Как работает паровой двигатель?",
    "Что такое инерционное звено первого порядка?",
]

for question in questions:
    response = llm.invoke(
        messages + [HumanMessage(content=question)]
    )

    answer = json.loads(response.content)

    assert set(answer) == {"short_answer", "full_answer"}
    assert all(isinstance(value, str) and value.strip()
               for value in answer.values())

    print(f"\nВопрос: {question}")
    print("Краткий ответ:", answer["short_answer"])
    print("Полный ответ:", answer["full_answer"])
    print("-" * 60)

print("\nВсе ответы прошли проверку формата.")
