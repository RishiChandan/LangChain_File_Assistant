# LangChain File Assistant (Agentic AI Demo)

This is a local file assistant project built using **LangChain**, **Jina embeddings**, and a **locally hosted LLaMA model via Ollama**.

## 🔍 Project Overview
The assistant allows you to upload PDFs, generate embeddings, and query documents using a conversational agent. It's built to demonstrate practical knowledge of agentic AI, tool usage, and multi-modal retrieval-augmented generation (RAG).

## 🧠 Features
- Document parsing and chunking
- Jina embedding-based vector search
- FAISS vectorstore for local persistence
- LangChain RetrievalQA agent with LLaMA via Ollama
- Console-based question answering loop

## 🛠️ Setup Instructions

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Start Ollama** and pull a model:
   ```
   ollama run llama3
   ```

3. **Add your JINA_API_KEY** to a `.env` file:
   ```
   JINA_API_KEY=your_key_here
   ```

4. **Place your PDF** in the `data/` folder as `sample.pdf`.

5. **Run the app pipeline**:
   ```
   python agent/app.py
   ```

6. **Query the assistant**:
   ```
   python agent/ask_agent.py
   ```

## 📄 Report
Detailed project report is available in `/reports/Agentic_File_Assistant_Report.pdf`.

## 🎥 Demo

![Demo of File Assistant](reports/sample_output.gif)


## 🚀 Future Improvements
- Web UI for uploading and querying
- Image understanding capabilities
- OpenAI or Together.ai integration for cloud demo

---

Built with ❤️ for agentic AI learning and demonstration.
