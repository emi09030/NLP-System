from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_DIR = "vectorstore/faiss_fit_hcmute"


def format_context(docs):
    parts = []

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")
        url = doc.metadata.get("url", "")
        category = doc.metadata.get("category", "")

        parts.append(
            f"[Tài liệu {i}]\n"
            f"Tiêu đề: {title}\n"
            f"Nguồn: {source}\n"
            f"URL: {url}\n"
            f"Nhóm: {category}\n"
            f"Nội dung: {doc.page_content}"
        )

    return "\n\n".join(parts)


def answer_without_llm(question, docs):
    context = format_context(docs)

    answer = f"""
Câu hỏi: {question}

Các đoạn dữ liệu liên quan nhất tìm được:

{context}
"""

    return answer.strip()


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    print("RAG Chatbot Khoa CNTT HCMUTE")
    print("Gõ 'exit' để thoát")

    while True:
        question = input("\nNhập câu hỏi: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        docs = retriever.invoke(question)

        answer = answer_without_llm(question, docs)

        print("\n" + answer)


if __name__ == "__main__":
    main()