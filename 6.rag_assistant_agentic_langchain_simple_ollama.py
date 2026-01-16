
import os
import logging
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever 
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# suppress all warnings
warnings.filterwarnings("ignore")

# Suppress ChromaDB warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.db.duckdb").setLevel(logging.ERROR)

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Custom prompt ----
prompt_template = PromptTemplate(
    template="""
You are a medical records assistant analyzing SYNTHETIC DEMO DATA for testing purposes.
This is NOT real patient data - it is artificially generated for demonstration.
Answer the user's question using only the patient records provided.
Do NOT repeat long chunks of text.
If the answer is not in the records, say: "Not found in the records."
Context:
{context}
Question: {question}
Answer clearly and directly:
""",
    input_variables=["context", "question"],
)

def build_hybrid_rag(pdf_path: str):
    # 1. load pdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 3a. semantic retriever with Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="mixtral",
        base_url="http://localhost:11434"
    )
    
    #  # 3a. semantic retriever (faiss)
    # embeddings = OpenAIEmbeddings()
    
    # 1. using FAISS or Chroma
    # vectorstore = FAISS.from_documents(docs, embeddings)
    # 2.using Chroma
    # vectorstore = Chroma.from_documents(docs, embeddings)
    # semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Using Chroma (you can also use FAISS)
    vectorstore = Chroma.from_documents(docs, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3b. keyword retriever (bm25)
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 3

    # 3c. combine them into a hybrid retriever
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.6, 0.4]  # semantic prioritized, but keywords matter
    )

    # 4. wrap retriever as a tool
    retrieval_pdf_tool = Tool(
        name="PatientRecordsSearch",
        func=retriever.get_relevant_documents,
        description="Look up patient information from medical PDF records."
    )
    
    # 5. define the agent with Ollama
    llm = Ollama(
        model="mixtral",
        temperature=0.2,
        base_url="http://localhost:11434"
    )
    # The issue is that Ollama's Llama 3.1 model has safety guardrails 
    # preventing it from discussing "private medical records" even though 
    # this is synthetic demo data. It's also struggling with the agent format. Let's fix this with multiple solutions:
    
    # Note: OPENAI_FUNCTIONS agent type won't work with Ollama
    # Use ZERO_SHOT_REACT_DESCRIPTION instead
    agent = initialize_agent(
        tools=[retrieval_pdf_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False, 
        handle_parsing_errors=True
    )
    return agent, retriever

if __name__ == "__main__":
    print("\n--- Agentic RAG Assistant with Ollama ---")
    
    pdf_path = os.getenv('PDF_PATH_FILE')
    
    try:
        agent, retriever = build_hybrid_rag(pdf_path)

        question = "What medications is Hassan Kim currently prescribed?"
        response = agent.invoke(question)
        print("\n" + "="*60)
        print("Question:", response["input"])
        print("Answer:", response["output"])
        print("="*60)
        
        source_documents = retriever.get_relevant_documents(question)    
        print(f"\nRelevant Sources ({len(source_documents)} documents found):")
        for i, doc in enumerate(source_documents, 1):
            source_file = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", "Unknown")
            # extract filename from full path
            if "\\" in source_file:
                filename = source_file.split("\\")[-1]
            elif "/" in source_file:
                filename = source_file.split("/")[-1]
            else:
                filename = source_file
            print(f"{i}. File: {filename}")
            print(f"   Page: {page_num}")
            print(f"   Content preview: {doc.page_content[:150]}...")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Ollama is running: ollama list")
        print("2. Download model: ollama pull llama3.1")
        print("3. Verify PDF path in .env file")
        print("4. Check if all packages are installed:")
        print("   pip install langchain langchain-community chromadb pypdf")