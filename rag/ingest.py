"""
Loads knowledge_base/content/*.md, attaches source metadata from
knowledge_base/sources.json, chunks the text, embeds it, and persists
everything into a local Chroma vector store.

Run once (and again any time the source content changes):
    python ingest.py
"""

import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
SOURCES_FILE = BASE_DIR / "sources.json"
CONTENT_DIR = BASE_DIR / "content"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents() -> list[Document]:
    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    docs = []

    for entry in sources:
        file_path = CONTENT_DIR / entry["file"]
        if not file_path.exists():
            print(f"WARNING: {file_path} not found, skipping.")
            continue

        raw = file_path.read_text(encoding="utf-8")
        # Strip the TITLE/URL/RETRIEVED header block if present (content after '---')
        body = raw.split("---", 1)[-1].strip() if "---" in raw else raw.strip()

        if len(body) < 100:
            print(f"WARNING: {file_path} has very little content ({len(body)} chars) — check it.")

        docs.append(Document(
            page_content=body,
            metadata={
                "source_title": entry["title"],
                "source_url": entry["url"],
            },
        ))
        print(f"Loaded: {entry['title']} ({len(body)} chars)")

    return docs


def main():
    documents = load_documents()
    if not documents:
        print("No documents loaded — check knowledge_base/content/. Aborting.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks.")

    print(f"Embedding with {EMBEDDING_MODEL} (first run downloads the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectorstore.persist()
    print(f"\nDone. Persisted {len(chunks)} chunks to {PERSIST_DIR}")

    # Quick sanity check
    print("\n--- Sanity check ---")
    results = vectorstore.similarity_search("must-visit attractions in Singapore", k=2)
    for r in results:
        print(f"- [{r.metadata['source_title']}] {r.page_content[:100]}...")


if __name__ == "__main__":
    main()
