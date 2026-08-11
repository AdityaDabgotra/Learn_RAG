# Learn_RAG

A practice repo for learning Retrieval-Augmented Generation (RAG) — building the retrieve → augment → generate pipeline from scratch, including document loading, chunking, embeddings, and vector storage with Chroma.

## Contents

- **notebook/** – Jupyter notebooks used to experiment with RAG concepts step by step
- **CampusX/** – Notes/code following CampusX RAG tutorials
- **data/** – Sample data used for retrieval experiments
- **chroma_db/** – Local Chroma vector database storage
- **practice.py** – Scratch script for testing RAG functions
- **PDF files** (`01_The_Lightning_Thief.pdf`, `gunaho ka devta.pdf`, etc.) – Sample documents used as the knowledge base for retrieval
- **requirements.txt** – Python dependencies for the project

## Core Functions

- `rag_retrieve()` – Retrieve relevant documents from a knowledge base based on a query
- `rag_generate()` – Generate a response based on the retrieved documents and the query
- `rag()` – High-level function combining retrieval and generation in one step

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/AdityaDabgotra/Learn_RAG.git
   cd Learn_RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in `notebook/` or explore `practice.py` to see the RAG pipeline in action.

## Purpose

This repo is for personal learning — understanding how RAG pipelines work, from document ingestion and chunking to embedding, vector storage, and retrieval-augmented generation.

## License

No license specified yet.
