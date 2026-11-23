import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "results" / "chunks.json"
DB = ROOT / "db" / "chroma"

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks():
    if not CHUNKS.exists():
        raise SystemExit("Сначала выполните: python src/rag_lab.py prepare")

    return json.loads(CHUNKS.read_text(encoding="utf-8"))


def documents(chunks):
    from langchain_core.documents import Document

    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "idx": chunk["idx"],
                "source": chunk["source"],
                "page": chunk["page"],
            },
        )
        for chunk in chunks
    ]


def embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=MODEL)


def vector_db():
    from langchain_chroma import Chroma

    if not (DB / "chroma.sqlite3").exists():
        raise SystemExit("Сначала выполните: python src/rag_lab.py index")

    return Chroma(
        collection_name="lr5",
        embedding_function=embeddings(),
        persist_directory=str(DB),
    )


# 1. PDF -> текст -> чанки
def prepare():
    import fitz
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pdfs = sorted((ROOT / "materials").glob("*.pdf"))

    if not pdfs:
        raise SystemExit("Положите PDF в каталог materials/")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    preview = []
    extracted = []
    total_pages = 0

    for pdf in pdfs:
        with fitz.open(pdf) as doc:
            total_pages += len(doc)

            for page_no, page in enumerate(doc, start=1):
                if page_no < 9:
                    continue
                raw = page.get_text("text", sort=True)

                # Убираем переносы слов и лишние пробелы
                text = re.sub(r"-\s*\n(?=\w)", "", raw)
                text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()

                if len(text) < 80:
                    continue

                extracted.append(
                    f"\n=== {pdf.name}, стр. {page_no} ===\n{text}\n"
                )

                for block in splitter.split_text(text):
                    idx = len(chunks)

                    chunks.append({
                        "idx": idx,
                        "source": pdf.name,
                        "page": page_no,
                        "text": block,
                    })

                    preview.append(
                        f"===== ЧАНК {idx} | {pdf.name} | "
                        f"стр. {page_no} =====\n{block}\n"
                    )

    if total_pages < 80:
        raise SystemExit(
            f"Всего {total_pages} страниц, требуется минимум 80"
        )

    if not chunks:
        raise SystemExit("Текст не извлечён. Проверьте PDF.")

    (ROOT / "results").mkdir(exist_ok=True)

    CHUNKS.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (ROOT / "results" / "extracted_text.txt").write_text(
        "\n".join(extracted),
        encoding="utf-8",
    )

    (ROOT / "results" / "chunks_preview.txt").write_text(
        "\n".join(preview),
        encoding="utf-8",
    )

    print(f"PDF-файлов: {len(pdfs)}")
    print(f"Страниц: {total_pages}")
    print(f"Чанков: {len(chunks)}")
    print("Результат: results/chunks_preview.txt")


# 2. Создание векторной базы
def index():
    import shutil
    from langchain_chroma import Chroma

    docs = documents(load_chunks())

    # Пересоздаём только локальную БД этой лабораторной
    if DB.exists():
        shutil.rmtree(DB)

    db = Chroma(
        collection_name="lr5",
        embedding_function=embeddings(),
        persist_directory=str(DB),
    )

    for start in range(0, len(docs), 100):
        batch = docs[start:start + 100]

        db.add_documents(
            batch,
            ids=[str(doc.metadata["idx"]) for doc in batch],
        )

        print(f"Проиндексировано: {min(start + 100, len(docs))}/{len(docs)}")


# 3. BM25
def tokenize(text):
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def bm25_db(docs):
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(
        docs,
        preprocess_func=tokenize,
    )


# 4. Dense, sparse и гибридный RRF-поиск
def search(query, k, method, db, bm25):
    if method == "dense":
        return db.similarity_search(query, k=k)

    if method == "sparse":
        bm25.k = k
        return bm25.invoke(query)

    if method == "hybrid":
        pool = max(20, k)

        dense = db.similarity_search(query, k=pool)

        bm25.k = pool
        sparse = bm25.invoke(query)

        scores = {}
        matched = {}

        for ranked in (dense, sparse):
            for rank, doc in enumerate(ranked, start=1):
                idx = doc.metadata["idx"]

                # Reciprocal Rank Fusion
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
                matched[idx] = doc

        order = sorted(
            scores,
            key=lambda idx: (-scores[idx], idx),
        )

        return [matched[idx] for idx in order[:k]]

    raise ValueError(f"Неизвестный метод: {method}")


# 5. Метрики качества
def metrics(found, gold, k):
    relevant = set(gold)
    retrieved = [doc.metadata["idx"] for doc in found[:k]]

    hits = len(relevant.intersection(retrieved))

    recall = hits / len(relevant)
    precision = hits / k

    mrr = next(
        (
            1.0 / rank
            for rank, idx in enumerate(retrieved, start=1)
            if idx in relevant
        ),
        0.0,
    )

    return recall, precision, mrr


# 6. Сравнение методов
def evaluate():
    import csv
    import sys

    sys.path.insert(0, str(ROOT))
    from test_queries import test_queries

    if len(test_queries) < 15:
        raise SystemExit("Нужно заполнить минимум 15 вопросов")

    if sum(len(q["golden_chunks"]) > 1 for q in test_queries) < 8:
        raise SystemExit(
            "Минимум 8 вопросов должны иметь больше одного золотого чанка"
        )

    valid_ids = {chunk["idx"] for chunk in load_chunks()}

    for q in test_queries:
        if (
            not q["golden_chunks"]
            or not set(q["golden_chunks"]) <= valid_ids
        ):
            raise SystemExit(f"Неверные golden_chunks: {q['query']}")

    db = vector_db()
    bm25 = bm25_db(documents(load_chunks()))

    output = []

    for method in ("dense", "sparse", "hybrid"):
        per_query = [
            search(q["query"], 5, method, db, bm25)
            for q in test_queries
        ]

        for k in (1, 3, 5):
            values = [
                metrics(found, q["golden_chunks"], k)
                for found, q in zip(per_query, test_queries)
            ]

            row = {
                "method": method,
                "k": k,
                "recall": round(
                    sum(x[0] for x in values) / len(values), 4
                ),
                "precision": round(
                    sum(x[1] for x in values) / len(values), 4
                ),
                "mrr": round(
                    sum(x[2] for x in values) / len(values), 4
                ),
            }

            output.append(row)
            print(row)

    with (ROOT / "results" / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "k", "recall", "precision", "mrr"],
        )
        writer.writeheader()
        writer.writerows(output)

    # Выбираем лучший метод по Recall@5,
    # затем по MRR@5 и Precision@5
    best = max(
        (row for row in output if row["k"] == 5),
        key=lambda row: (
            row["recall"], row["mrr"], row["precision"]
        ),
    )["method"]

    (ROOT / "results" / "best_method.txt").write_text(
        best, encoding="utf-8"
    )

    print("\nЛучший метод при k=5:", best)


# 7. Финальный RAG
def ask(question, method=None):
    from langchain_ollama import ChatOllama

    if method is None:
        path = ROOT / "results" / "best_method.txt"

        if not path.exists():
            raise SystemExit(
                "Сначала выполните: python src/rag_lab.py evaluate"
            )

        method = path.read_text(encoding="utf-8").strip()

    db = vector_db()
    bm25 = bm25_db(documents(load_chunks()))

    found = search(question, 5, method, db, bm25)

    context = "\n\n".join(
        f"Фрагмент {doc.metadata['idx']}: {doc.page_content}"
        for doc in found
    )

    llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11435",
    temperature=0,
    format={
        "type": "object",
        "properties": {
            "short_answer": {
                "type": "string",
                "description": "Краткий ответ на вопрос",
            },
            "full_answer": {
                "type": "string",
                "description": "Подробный ответ на вопрос",
            },
        },
        "required": [
            "short_answer",
            "full_answer",
        ],
        "additionalProperties": False,
    },
)

    instruction = (
        "Отвечай исключительно по предоставленным фрагментам, "
        "без использования внешних знаний. "
        "Верни только JSON с полями short_answer и full_answer. "
        "Краткий ответ — несколько слов, полный — несколько предложений. "
        "Если в фрагментах нет достоверного ответа, укажи "
        'short_answer="Недостаточно данных", '
        'full_answer="Данных для ответа недостаточно". '
        "Не создавай список источников, он будет добавлен программой.\n\n"
        f"Фрагменты:\n{context}\n\nВопрос: {question}"
    )

    response = llm.invoke(instruction)

    try:
        answer = json.loads(response.content)

        short = answer["short_answer"]
        full = answer["full_answer"]

        if not isinstance(short, str) or not isinstance(full, str):
            raise ValueError("Ответы должны быть строками")

        if not short.strip() or not full.strip():
            raise ValueError("Ответы не должны быть пустыми")

    except (ValueError, KeyError, TypeError) as exc:
        print("Некорректный ответ модели:")
        print(response.content)

        raise SystemExit(f"Ошибка формата ответа LLM: {exc}") from exc

    # Нормализация ответа при недостатке информации
    if short.strip().lower() == "недостаточно данных":
        full = "Данных для ответа недостаточно"

    print(f"\nМетод поиска: {method}")
    print(f"Краткий ответ: {short}")
    print(f"Полный ответ: {full}")

    # Список источников формирует Python, а не LLM
    print("\nИсточники:")

    for doc in found:
        print(
            f"- [{doc.metadata['idx']}] "
            f"{doc.metadata['source']}, стр. {doc.metadata['page']}"
        )
        print(" ", doc.page_content[:250].replace("\n", " "), "...")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["prepare", "index", "search", "evaluate", "ask"],
    )

    parser.add_argument("query", nargs="?")

    parser.add_argument(
        "--method",
        choices=["dense", "sparse", "hybrid"],
    )

    parser.add_argument("-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare()

    elif args.command == "index":
        index()

    elif args.command == "evaluate":
        evaluate()

    elif args.command == "search":
        if not args.query:
            parser.error("Укажите поисковый запрос")

        method = args.method or "hybrid"
        db = vector_db()
        bm25 = bm25_db(documents(load_chunks()))

        for doc in search(args.query, args.k, method, db, bm25):
            print(
                f"[{doc.metadata['idx']}] "
                f"{doc.metadata['source']}, стр. {doc.metadata['page']}"
            )
            print(doc.page_content[:500], "\n")

    elif args.command == "ask":
        if not args.query:
            parser.error("Укажите вопрос")

        ask(args.query, args.method)


if __name__ == "__main__":
    main()
