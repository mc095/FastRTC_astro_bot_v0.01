import os
import time
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from groq import Groq

# Configuration
GROQ_API_KEY = "YOUR_GROQ_API"
HF_TOKEN = "YOUR_HF_TOKEN"
ASTRO_PROMPT = (
    "You are an engaging space buddy who provides very concise responses."
    "After delivering information, you end with an open-ended question to encourage further discussion. "
    "you initiate a new topic or ask a question to maintain the conversation. Listen to the user carefully. "
)

groq_client = Groq(api_key=GROQ_API_KEY)
os.environ["HF_TOKEN"] = HF_TOKEN
stt_model = get_stt_model()
tts_model = get_tts_model()

def astro_talk(audio):
    """Process audio input and stream response with latency measurement."""
    start_time = time.time()
    
    # Speech-to-text
    user_input = stt_model.stt(audio)
    print(f"User input: {user_input}") 
    
    # Groq response
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": ASTRO_PROMPT},
            {"role": "user", "content": user_input}
        ],
        max_tokens=200,
        temperature=0.7,
        stream=False
    )
    
    astro_reply = response.choices[0].message.content
    latency = time.time() - start_time
    print(f"Response latency: {latency:.3f} seconds")
    
    # Stream audio response
    for audio_chunk in tts_model.stream_tts_sync(astro_reply):
        yield audio_chunk

# Stream with voice detection and 3-second timeout
stream = Stream(
    ReplyOnPause(astro_talk),
    modality="audio",
    mode="send-receive"
)

# Launch Gradio UI
stream.ui.launch(
    inbrowser=True,
    server_port=7860,
    quiet=True
)
