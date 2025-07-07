import os
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from rich.console import Console

# Load .env variables
load_dotenv()
jina_api_key = os.getenv("JINA_API_KEY")

# Load FAISS vector store and retriever
def load_vectorstore(folder="vectorstore"):
    embedding_model = JinaEmbeddings(
        jina_api_key=jina_api_key,
        model_name="jina-embeddings-v2-base-en"
    )
    return FAISS.load_local(folder, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Create a QA chain using Ollama and retriever
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = OllamaLLM(model="llama3")

    prompt_template = '''
You are a helpful assistant. Use the following extracted parts from a PDF document to answer the user's question.
If the answer is not in the document, say so clearly. Be specific and detailed.

Context:
{context}

Question:
{question}
'''
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    qa_chain = create_qa_chain(vectorstore)
    console = Console()

    while True:
        query = input("\n❓ Your question (or type 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        with console.status("[bold green]🤖 Thinking... Please wait...", spinner="dots"):
            answer = qa_chain.invoke({"query": query})

        console.rule("[bold blue]🧠 Answer")
        console.print(answer["result"].strip(), style="green")
        console.rule()
