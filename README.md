# Rag-application


# 📄 DocuMind – Chat with your PDF

A simple RAG (Retrieval-Augmented Generation) app built with **LlamaIndex** and **Streamlit**.
Upload a PDF, ask questions in plain language, and get answers based on your document, along with the source passages used.

---

## ✨ Features

- Upload any PDF and chat with it
- Answers grounded in your document (no guessing)
- See the exact source passages, page numbers and similarity scores
- Adjustable retrieval settings (top-k and similarity cutoff)
- Clean UI with light and dark mode support

---

## 🧠 How It Works

```
PDF → Read pages → Chunk + Embed → Vector Index → Retrieve top-k → Filter by similarity → LLM → Answer
```

1. **Load**: `PyMuPDFReader` reads the PDF.
2. **Index**: LlamaIndex splits the text into chunks and embeds them with `all-MiniLM-L6-v2`.
3. **Retrieve**: `VectorIndexRetriever` finds the most relevant chunks for your question.
4. **Filter**: `SimilarityPostprocessor` removes chunks below the similarity cutoff.
5. **Generate**: the LLM (via OpenRouter) writes the answer from the remaining chunks.

---

## 🛠️ Tech Stack

| Part | Tool |
|------|------|
| UI | Streamlit |
| RAG framework | LlamaIndex |
| PDF reader | PyMuPDF |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| LLM | `openai/gpt-oss-20b` via OpenRouter |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Sourav1000888/Rag-application.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project folder:

```
OPENROUTER_API_KEY=your_api_key_here
```

Get a key at [openrouter.ai](https://openrouter.ai). You can also paste it into the app's sidebar instead.

### 4. Run the app

```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`).

---

## 💡 Usage

1. Enter your API key in the sidebar (if not using `.env`).
2. Upload a PDF.
3. Click **Index document**.
4. Ask questions in the chat box.
5. Open **Sources** under any answer to see where it came from.

---

## ⚙️ Settings

| Setting | Default | What it does |
|---------|---------|--------------|
| LLM model | `openai/gpt-oss-20b` | Any model available on OpenRouter |
| Top-k | 4 | Number of passages retrieved per question |
| Similarity cutoff | 0.40 | Passages scoring below this are dropped |

**Tip:** if you get "no relevant information" answers, lower the similarity cutoff or increase top-k.

---

## 📁 Project Structure

```
├── app.py              # Streamlit app
├── first.ipynb         # Original notebook (RAG prototype)
├── requirements.txt    # Dependencies
├── .env                # API key (not committed)
└── README.md
```

---

## 🔮 Future Improvements

- Persistent vector store (Chroma / FAISS / Pinecone)
- Support for multiple PDFs and other file types
- Streaming responses
- Chat memory for follow-up questions

---

## ⚠️ Notes

- The index lives in memory, so you need to re-index after restarting the app.
- Never commit your `.env` file. Add it to `.gitignore`.


