import os
import asyncio
import edge_tts
from openai import OpenAI
import config
from dashscope.audio.asr import Recognition

class VoiceBot:
    def __init__(self):
        # Initialize OpenAI client only if key exists
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        self.dashscope_key = config.DASHSCOPE_API_KEY

    def speech_to_text(self, audio_path):
        """
        Transcribe audio file to text.
        Priority: DashScope (Qwen/Paraformer) > OpenAI Whisper
        """
        if not audio_path:
            return ""

        # 1. Try DashScope (Alibaba)
        if self.dashscope_key:
            try:
                rec = Recognition(model='paraformer-realtime-v1', format='wav', sample_rate=16000, callback=None)
                # DashScope SDK usage for file transcription might differ, using simple call here
                # Simplified: Using DashScope's call method for short audio
                # Note: DashScope python SDK might require different usage.
                # Let's use the simplest 'paraformer-v1' sync call
                recognition = Recognition(model='paraformer-v1', format='wav', sample_rate=16000)
                # Paraformer requires specific formats usually.
                # For robustness, we might want to just assume the file is compatible or use a conversion tool.
                # Assuming input is wav/mp3 compatible.
                result = recognition.call(audio_path)
                if result.status_code == 200:
                    # Extract text
                    sentences = [s['text'] for s in result.get_sentence()]
                    return "".join(sentences)
                else:
                    print(f"DashScope ASR Error: {result.message}")
            except Exception as e:
                print(f"Error in DashScope ASR: {e}")

        # 2. Try OpenAI Whisper (Fallback)
        if self.openai_client:
            try:
                with open(audio_path, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                return transcript.text
            except Exception as e:
                return f"Error in OpenAI ASR: {str(e)}"
        
        return "Error: No valid ASR service available (Check DASHSCOPE_API_KEY or OPENAI_API_KEY)."

    def text_to_speech(self, text, output_path="output.mp3"):
        """
        Convert text to speech.
        Priority: Edge-TTS (Free, High Quality) > OpenAI TTS
        """
        
        # 1. Try Edge-TTS (Free, works locally/network)
        try:
            # Create a loop to run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            loop.run_until_complete(communicate.save(output_path))
            return output_path
        except Exception as e:
            print(f"Error in Edge-TTS: {e}")

        # 2. Try OpenAI TTS (Fallback)
        if self.openai_client:
            try:
                response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text
                )
                response.stream_to_file(output_path)
                return output_path
            except Exception as e:
                print(f"Error in OpenAI TTS: {e}")
                
        return None
