import gradio as gr
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from crawler import WebCrawler
from rag import KnowledgeBase
from voice import VoiceBot

# Initialize modules
logger.info("Initializing modules...")
crawler = WebCrawler()
logger.info("WebCrawler initialized")
kb = KnowledgeBase()
logger.info("KnowledgeBase initialized")
voice_bot = VoiceBot()
logger.info("VoiceBot initialized")
logger.info("All modules initialized successfully")

def process_url(url):
    """
    Crawl URL and add to knowledge base
    """
    if not url:
        return "Please enter a URL."
    
    status_msg = f"Crawling {url}..."
    yield status_msg
    
    text, filepath = crawler.crawl(url)
    if not text:
        yield f"Failed to crawl {url}"
        return

    status_msg = f"Crawled successfully. Indexing..."
    yield status_msg
    
    num_chunks = kb.add_document(filepath)
    yield f"Indexed {num_chunks} chunks from {url}. Ready to query."

def chat_response(message, history):
    """
    Chatbot response function
    """
    if not message:
        return ""
    response = kb.query(message)
    return response

def voice_interaction(audio_path):
    """
    Voice -> Text -> RAG -> Text -> Audio
    """
    if not audio_path:
        return None, "No audio provided."
    
    # 1. ASR
    user_text = voice_bot.speech_to_text(audio_path)
    if not user_text:
        return None, "Could not transcribe audio."
    
    # 2. RAG
    bot_text = kb.query(user_text)
    
    # 3. TTS
    output_audio_path = f"response_{os.path.basename(audio_path)}.mp3"
    saved_path = voice_bot.text_to_speech(bot_text, output_path=output_audio_path)
    
    return saved_path, f"User: {user_text}\nBot: {bot_text}"

# Build Gradio UI
with gr.Blocks(title="Knowledge Bot") as demo:
    gr.Markdown("# 🤖 Knowledge Bot with Voice & RAG")
    
    with gr.Tab("Knowledge Base"):
        gr.Markdown("### Add Knowledge from URL")
        url_input = gr.Textbox(label="Website URL", placeholder="https://example.com")
        crawl_btn = gr.Button("Crawl & Index")
        status_output = gr.Textbox(label="Status", interactive=False)
        
        crawl_btn.click(process_url, inputs=[url_input], outputs=[status_output])

    with gr.Tab("Chatbot"):
        gr.Markdown("### Chat with your Knowledge Base")
        chat_interface = gr.ChatInterface(fn=chat_response)

    with gr.Tab("Voice Bot"):
        gr.Markdown("### Talk to your Knowledge Base")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak")
                submit_voice_btn = gr.Button("Send Voice")
            with gr.Column():
                audio_output = gr.Audio(label="Response Audio")
                transcript_output = gr.Textbox(label="Transcript")
        
        submit_voice_btn.click(
            voice_interaction, 
            inputs=[audio_input], 
            outputs=[audio_output, transcript_output]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
