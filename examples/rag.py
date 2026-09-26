"""Minimal RAG with ollamactl: embeddings with bge-m3, answers with qwen2.5:7b.

    python examples/rag.py

Swap DOCS for your own chunks. Past a few thousand chunks, keep the vectors in a
vector database (Chroma, sqlite-vec, pgvector...) instead of a Python list.
"""
import math

from ollamactl import Ollama

EMBED_MODEL = "bge-m3"     # multilingual embedding model
CHAT_MODEL = "qwen2.5:7b"  # needs the "tools" capability for the agentic mode

DOCS = [
    "Ollama runs large language models locally, without depending on the cloud.",
    "Recife is the capital of Pernambuco, on the northeastern coast of Brazil.",
    "Photosynthesis turns sunlight into chemical energy in plants.",
    "The store's refund window is 7 calendar days after delivery.",
]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


with Ollama() as ai:
    ai.load(EMBED_MODEL)
    ai.load(CHAT_MODEL)
    vectors = ai.embed(EMBED_MODEL, DOCS)["embeddings"]  # send large corpora in batches

    def search(query: str, k: int = 2) -> list:
        "Searches the knowledge base for the passages most relevant to the query."
        qv = ai.embed(EMBED_MODEL, query)["embeddings"][0]
        ranked = sorted(zip(DOCS, vectors), key=lambda p: cosine(qv, p[1]), reverse=True)
        return [doc for doc, _ in ranked[:k]]

    # num_ctx: Ollama silently drops the start of prompts longer than the context window
    options = {"num_ctx": 8192, "temperature": 0}

    # 1) classic RAG: retrieve, put the passages in the prompt, answer
    question = "How many days do I have to ask for a refund?"
    context = "\n".join(search(question))
    r = ai.chat(CHAT_MODEL, [
        {"role": "system", "content": f"Answer using only this context:\n{context}"},
        {"role": "user", "content": question},
    ], options=options)
    print("classic:", r["message"]["content"])

    # 2) agentic RAG: the model decides when, and what, to search
    msgs = [{"role": "user", "content": "What is the capital of Pernambuco? Check the knowledge base."}]
    r = ai.chat_tools(CHAT_MODEL, msgs, tools=[search], options=options)
    print("searched for:", [c["function"]["arguments"] for m in msgs for c in m.get("tool_calls", [])])
    print("agentic:", r["message"]["content"])
