import gradio as gr
from src.etl.cleaner import DocumentCleaner
from src.database.chunker import SemanticChunker
from src.database.vector_db import VectorDatabaseManager
from src.rag.retriever import RAGRetriever
from src.rag.llm_twin_generator import LLMTwinGenerator

# 1. Initialize Pipeline & Data Ingestion on Startup
print("[INFO] Initializing LLM Twin Pipeline...")
cleaner = DocumentCleaner()
docs = cleaner.load_and_clean_raw_posts()

chunker = SemanticChunker()
chunks = chunker.chunk_documents(docs)

vector_db = VectorDatabaseManager()
vector_db.index_chunks(chunks)

retriever = RAGRetriever(vector_db)
generator = LLMTwinGenerator(retriever)

# 2. Gradio Response Handler
def respond(user_message, history):
    if not user_message or not user_message.strip():
        return history, ""
        
    res = generator.generate_response(user_message)
    answer = res["answer"]
    
    if res["sources"]:
        sources_str = "\n\n---\n**📚 Context Retrieved from Past Writing:**\n"
        for s in res["sources"]:
            sources_str += f"- [{s['title']}]({s['source_url'] or '#'}) (Relevance: `{s['score']}`)\n"
        answer += sources_str
        
    history.append((user_message, answer))
    return history, ""

def populate_prompt(example_text):
    return example_text

# 3. Modern Custom CSS & Design System
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    color: #f8fafc !important;
}

.hero-banner {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.hero-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}

.chat-box {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(10px);
}

.input-container {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 2px solid #6366f1 !important;
    border-radius: 12px !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
}

.send-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4) !important;
}

footer { visibility: hidden !important; }
"""

with gr.Blocks(title="LLM Twin - AI Personal Assistant") as demo:
    
    # Hero Header Banner
    gr.HTML(
        """
        <div class="hero-banner">
            <div class="hero-title">🤖 LLM Twin — AI Personal Character Assistant</div>
            <div style="color: #94a3b8; font-size: 15px;">
                An intelligent AI twin trained on personal technical writing, grounded by <b>Qdrant Vector RAG</b>, and served via <b>Groq Llama 3</b>.
            </div>
            <div class="badge-row">
                <span class="badge">⚡ Groq Llama 3.3 70B</span>
                <span class="badge">🔍 Qdrant Vector Cloud</span>
                <span class="badge">🧠 QLoRA Fine-Tuned Adapter</span>
                <span class="badge">✅ LLM-as-a-Judge Gate Passed</span>
            </div>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            # Prominent Interactive Chatbot Area
            chatbot = gr.Chatbot(
                height=450,
                elem_classes=["chat-box"],
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=LLMTwin")
            )
            
            # Prominent Input Box Row
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="💬 Type your message or question here... (Press Enter to Send)",
                    container=True,
                    scale=5,
                    show_label=False,
                    elem_classes=["input-container"]
                )
                submit_btn = gr.Button("Send 🚀", scale=1, elem_classes=["send-btn"])
            
            # Sample Quick Questions Chips
            gr.Markdown("#### 💡 Quick Prompt Suggestions (Click to Ask):")
            with gr.Row():
                btn1 = gr.Button("⚡ How do you build production RAG?", size="sm")
                btn2 = gr.Button("🧠 Explain your QLoRA Fine-Tuning workflow", size="sm")
                btn3 = gr.Button("📊 How does your LLM-as-a-judge gate work?", size="sm")
                
            clear = gr.Button("🗑️ Clear Chat History", size="sm", variant="secondary")

    # Wire up Event Handlers
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    
    btn1.click(populate_prompt, inputs=[btn1], outputs=[msg]).then(respond, [msg, chatbot], [chatbot, msg])
    btn2.click(populate_prompt, inputs=[btn2], outputs=[msg]).then(respond, [msg, chatbot], [chatbot, msg])
    btn3.click(populate_prompt, inputs=[btn3], outputs=[msg]).then(respond, [msg, chatbot], [chatbot, msg])
    
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft(), css=custom_css)
