import gradio as gr
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Multiple key formats (handles any secret issue)
api_keys = [
    os.getenv("openai_api_key"),
    os.getenv("OPENAI_API_KEY"), 
    os.getenv("OPENAI_APIKEY")
]

os.environ["OPENAI_API_KEY"] = next((key for key in api_keys if key and len(key) > 20), None)
print(f"Using API key: {'✅ VALID' if os.getenv('OPENAI_API_KEY') else '❌ MISSING'}")

try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="./legal_precedent_db", embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("✅ ChromaDB + OpenAI loaded!")
except Exception as e:
    print(f"❌ Load error: {e}")

def legal_query(question):
    if not os.getenv("OPENAI_API_KEY"):
        return "❌ OpenAI API key missing. Check Settings → Secrets → openai_api_key", "Setup required"
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(question)
        context = "\n\n".join([f"[{d.metadata.get('case_id', 'Unknown')}] {d.page_content[:300]}..." for d in docs])
        
        prompt = ChatPromptTemplate.from_template("""
        You are a legal research assistant. Cite Ohio cases with ¶ numbers.
        Context: {context}
        Question: {question}
        Answer with citations:
        """)
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        
        sources = [f"{d.metadata.get('case_id', 'Unknown')} (p{d.metadata['page']})" for d in docs[:3]]
        return response.content, "\n".join(sources)
    except Exception as e:
        return f"Error: {str(e)}", "Check OpenAI key and ChromaDB folder"

with gr.Blocks(title="Legal Precedent Analyzer") as demo:
    gr.Markdown("# ⚖️ Legal Precedent Analyzer")
    gr.Markdown("Ohio Supreme Court RAG - 10+ cases • ChromaDB 1,000+ chunks")
    
    with gr.Row():
        question = gr.Textbox(label="Legal Query", placeholder="Ohio liability damages tables", lines=2)
        submit = gr.Button("Search Precedents", variant="primary")
    
    with gr.Row():
        answer = gr.Textbox(label="Analysis", lines=10)
        sources = gr.Textbox(label="Cases Cited", lines=3)

    submit.click(legal_query, inputs=question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
