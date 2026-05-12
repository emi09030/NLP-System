from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


VECTOR_DIR = "vectorstore/faiss_fit_hcmute"


def format_docs(docs):
    parts = []

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "")
        source = doc.metadata.get("source", "")
        url = doc.metadata.get("url", "")
        content = doc.page_content

        parts.append(
            f"[Nguồn {i}]\n"
            f"Tiêu đề: {title}\n"
            f"File: {source}\n"
            f"URL: {url}\n"
            f"Nội dung: {content}"
        )

    return "\n\n".join(parts)


def build_answer(llm, question, docs):
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        """
Bạn là hệ thống hỏi đáp về Khoa Công nghệ Thông tin HCMUTE.
Chỉ trả lời dựa trên dữ liệu được cung cấp.
Nếu dữ liệu không có thông tin, hãy nói: "Tôi chưa tìm thấy thông tin này trong dữ liệu hiện có."
Trả lời bằng tiếng Việt, rõ ràng, ngắn gọn.

Dữ liệu:
{context}

Câu hỏi:
{question}

Câu trả lời:
"""
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return result.content


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

    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.2,
    )

    print("RAG Chatbot Khoa CNTT HCMUTE")
    print("Gõ 'exit' để thoát")

    while True:
        question = input("\nNhập câu hỏi: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        docs = retriever.invoke(question)
        answer = build_answer(llm, question, docs)

        print("\nTrả lời:")
        print(answer)

        print("\nNguồn tham khảo:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata.get('title', '')} - {doc.metadata.get('source', '')}")


if __name__ == "__main__":
    main()