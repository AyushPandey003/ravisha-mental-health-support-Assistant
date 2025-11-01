"""
Arnish - Professional Mental Health AI Assistant
FastAPI WebSocket Mental Health Assistant - Ultra Low Latency
Optimized for edge deployment with streaming capabilities
"""
import os
import asyncio
import json
import base64
import tempfile
import numpy as np
import librosa
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import google.generativeai as genai
from gtts import gTTS

import soundfile as sf

SAMPLE_RATE = 16000
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB1YM3JUNwe9BtenrBYI0P2NaNQFQzVvEY")

app = FastAPI(title="Arnish - Mental Health AI Assistant API")

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai_client = None


async def load_models():
    """Initialize API clients on startup"""
    global genai_client
    
    print("[startup] Initializing Gemini client for transcription and AI responses")
    genai.configure(api_key=GOOGLE_API_KEY)
    genai_client = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("[startup] Gemini client ready")


@app.on_event("startup")
async def startup_event():
    await load_models()




async def transcribe_audio(audio_data: np.ndarray, language: Optional[str] = None) -> tuple[str, str]:
    """Transcribe audio using Gemini's audio understanding with improved Hindi detection"""
    if not genai_client:
        raise Exception("Gemini client not initialized.")
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_data, SAMPLE_RATE)
        tmp_path = tmp.name
    
    try:
        # Upload audio file to Gemini
        audio_file = genai.upload_file(path=tmp_path)
        
        # Determine language instruction
        if language and language != "auto":
            lang_instruction = "Transcribe this audio in Hindi (Devanagari script)" if language == "hi" else "Transcribe this audio in English"
        else:
            lang_instruction = "Transcribe this audio. Detect the language automatically. If it's Hindi, use Devanagari script. If it's English, use English."
        
        # Create transcription prompt
        prompt = f"""{lang_instruction}

Important:
- Output ONLY the transcribed text, nothing else
- Do not add any explanations, labels, or formatting
- If Hindi, use Devanagari script (देवनागरी)
- If English, use standard English text"""
        
        # Get transcription from Gemini
        response = genai_client.generate_content([audio_file, prompt])
        text = response.text.strip()
        
        # Detect language from transcription
        has_devanagari = text and any('\u0900' <= char <= '\u097F' for char in text)
        
        hindi_phonetic_patterns = [
            'kya', 'hai', 'hoon', 'mein', 'main', 'aap', 'tum', 'hum',
            'kaise', 'kahan', 'kab', 'kyun', 'nahin', 'nahi', 'thik',
            'accha', 'theek', 'bahut', 'bohot', 'kuch', 'koi', 'yeh', 'woh'
        ]
        words = text.lower().split()
        hindi_pattern_count = sum(1 for word in words if any(pattern in word for pattern in hindi_phonetic_patterns))
        
        detected_lang = "hi" if (has_devanagari or hindi_pattern_count >= 2) else "en"
        
        print(f"[transcribe] Detected language: {detected_lang}, Text: {text[:100]}...")
        
        # If auto-detect and appears to be Hindi but not in Devanagari, retry with Hindi forced
        if (not language or language == "auto") and hindi_pattern_count >= 2 and not has_devanagari:
            print(f"[transcribe] Hindi detected, re-transcribing with Hindi forced...")
            
            prompt_hindi = """Transcribe this audio in Hindi using ONLY Devanagari script (देवनागरी लिपि).

Important:
- Output ONLY the Hindi transcription in Devanagari
- Do not use Roman/Latin script
- Do not add explanations"""
            
            response = genai_client.generate_content([audio_file, prompt_hindi])
            text = response.text.strip()
            detected_lang = "hi"
            print(f"[transcribe] Hindi transcription: {text[:100]}...")
        
        return text, detected_lang
        
    except Exception as e:
        print(f"[transcribe] Error: {str(e)}")
        return "", "en"
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def detect_crisis_keywords(text: str) -> bool:
    """Detect crisis keywords"""
    crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'self harm',
        'hurt myself', 'no reason to live', 'better off dead',
        'आत्महत्या', 'मरना चाहता', 'जान देना', 'खुद को नुकसान'
    ]
    return any(keyword in text.lower() for keyword in crisis_keywords)





async def get_ai_response(prompt: str, language: str = "en", max_retries: int = 3) -> str:
    """Get AI response from Gemini with retry logic"""
    lang_instruction = "Hindi (हिंदी)" if language == "hi" else "English"
    
    system_context = f"""You are Arnish, a compassionate and professional mental health support assistant specialized in providing emotional support and guidance keep your responses brief if some techniques were asked give best trending mental health tips in 10 to 15 sentences your response should be of minimum 3 sentence max 15.

Your role:
- Provide empathetic, supportive responses
- Offer practical coping strategies and techniques
- Help users understand their emotions
- Guide users through difficult situations
- Encourage healthy habits and positive thinking
- Suggest breathing exercises, mindfulness, and relaxation techniques when appropriate

Guidelines:
- Keep responses conversational and natural (7-8 sentences for voice interaction)
- Be warm, understanding, and non-judgmental
- Focus on mental wellness, stress management, anxiety relief, and emotional support
- Avoid giving medical advice or diagnoses
- In crisis situations, encourage professional help

CRITICAL Language Instruction: 
- If language is Hindi (hi), you MUST respond ONLY in Hindi using Devanagari script (देवनागरी लिपि).
- If language is English (en), respond only in English.
- Current language: {lang_instruction}
{f"- YOU MUST USE HINDI DEVANAGARI SCRIPT ONLY (जैसे: नमस्ते, मैं रविशा हूं)" if language == "hi" else ""}

User message: {prompt}

Your response:"""
    
    for attempt in range(max_retries):
        try:
            response = genai_client.generate_content(system_context)
            return response.text.replace('*', '').replace('#', '').replace('`', '')
        except Exception as e:
            error_msg = str(e)
            print(f"[gemini] Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            # Check if it's a 503 (overloaded) error
            if "503" in error_msg or "overloaded" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    print(f"[gemini] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
            
            # For other errors, return fallback immediately
            break
    
    # Fallback responses based on language
    if language.startswith('hi'):
        return "मुझे अभी आपकी बात समझने में परेशानी हो रही है। कृपया फिर से कोशिश करें।"
    return "I'm having trouble processing that right now. Please try again in a moment."


@app.get("/")
async def get_client():
    """Serve an improved WebSocket client UI"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Arnish - Your Mental Health AI Assistant</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 900px;
                width: 100%;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 28px;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }
            
            .header p {
                font-size: 14px;
                opacity: 0.9;
            }
            
            #status { 
                padding: 12px 20px;
                margin: 20px 30px;
                border-radius: 10px;
                text-align: center;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.3s;
            }
            
            .connected { 
                background: #d4edda; 
                color: #155724; 
                border: 2px solid #28a745;
            }
            
            .disconnected { 
                background: #f8d7da; 
                color: #721c24; 
                border: 2px solid #dc3545;
            }
            
            .controls {
                padding: 0 30px 20px 30px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .language-selector {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            
            .language-selector label {
                font-weight: 600;
                color: #333;
            }
            
            .language-selector select {
                flex: 1;
                padding: 10px;
                border: 2px solid #667eea;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                background: white;
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            
            button { 
                flex: 1;
                min-width: 150px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                border: none;
                border-radius: 10px;
                transition: all 0.3s;
                color: white;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            .btn-connect {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .btn-record {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            
            .btn-stop {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }
            
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            #messages { 
                padding: 20px 30px 30px 30px;
                height: 450px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            #messages::-webkit-scrollbar {
                width: 8px;
            }
            
            #messages::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            
            #messages::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 4px;
            }
            
            .message { 
                margin: 15px 0;
                padding: 15px 20px;
                border-radius: 15px;
                animation: slideIn 0.3s ease;
                max-width: 85%;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .user { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 5px;
            }
            
            .assistant { 
                background: white;
                color: #333;
                border: 2px solid #e9ecef;
                margin-right: auto;
                border-bottom-left-radius: 5px;
            }
            
            .system {
                background: #fff3cd;
                color: #856404;
                text-align: center;
                margin: 10px auto;
                max-width: 100%;
                font-size: 14px;
            }
            
            .error { 
                background: #f8d7da;
                color: #721c24;
                border: 2px solid #dc3545;
                text-align: center;
                margin: 10px auto;
                max-width: 100%;
            }
            
            .recording-indicator {
                display: none;
                align-items: center;
                gap: 10px;
                padding: 15px;
                background: #fff3cd;
                border-radius: 10px;
                margin: 0 30px 20px 30px;
            }
            
            .recording-indicator.active {
                display: flex;
            }
            
            .pulse {
                width: 12px;
                height: 12px;
                background: #dc3545;
                border-radius: 50%;
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                }
                
                .header h1 {
                    font-size: 22px;
                }
                
                button {
                    min-width: 120px;
                    padding: 12px 20px;
                    font-size: 14px;
                }
                
                #messages {
                    height: 350px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>
                    <span>🧠</span>
                    <span>Arnish</span>
                </h1>
                <p>Your Professional Mental Health AI Assistant - Compassionate support and guidance whenever you need it</p>
            </div>
            
            <div id="status" class="disconnected">● Disconnected - Click Connect to Start</div>
            
            <div class="recording-indicator" id="recordingIndicator">
                <div class="pulse"></div>
                <span>Recording... Speak clearly into your microphone</span>
            </div>
            
            <div class="controls">
                <div class="language-selector">
                    <label>🌐 Language:</label>
                    <select id="language">
                        <option value="en">English</option>
                        <option value="hi">हिंदी (Hindi)</option>
                        <option value="auto">Auto-detect</option>
                    </select>
                </div>
                
                <div class="button-group">
                    <button class="btn-connect" onclick="connect()">🔌 Connect</button>
                    <button class="btn-record" onclick="startRecording()" id="recordBtn" disabled>🎤 Start Recording</button>
                    <button class="btn-stop" onclick="stopRecording()" id="stopBtn" disabled>⏹️ Stop Recording</button>
                </div>
            </div>
            
            <div id="messages"></div>
        </div>
        
        <script>
            let ws = null;
            let mediaRecorder = null;
            let audioChunks = [];
            
            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    document.getElementById('status').textContent = '● Connected - Ready to Help';
                    document.getElementById('status').className = 'connected';
                    document.getElementById('recordBtn').disabled = false;
                    addMessage('system', '✓ Connected! Select your language and start recording to begin.');
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = () => {
                    document.getElementById('status').textContent = '● Disconnected - Click Connect';
                    document.getElementById('status').className = 'disconnected';
                    document.getElementById('recordBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = true;
                    addMessage('system', 'Connection closed. Please reconnect.');
                };
                
                ws.onerror = (error) => {
                    addMessage('error', 'Connection error. Please check your internet and try again.');
                };
            }
            
            async function startRecording() {
                try {
                    // Check if browser supports getUserMedia
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        const hostname = window.location.hostname;
                        const port = window.location.port || '8001';
                        throw new Error(`Microphone not available. Enable it in Chrome:\nchrome://flags/#unsafely-treat-insecure-origin-as-secure\nAdd: http://${hostname}:${port}`);
                    }
                    
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            channelCount: 1,
                            sampleRate: 16000,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true
                        } 
                    });
                    
                    // Try to use better codec if available
                    let options = { mimeType: 'audio/webm;codecs=opus' };
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        options = { mimeType: 'audio/webm' };
                    }
                    
                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        document.getElementById('recordingIndicator').classList.remove('active');
                        document.getElementById('recordBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const base64Audio = reader.result.split(',')[1];
                            const selectedLang = document.getElementById('language').value;
                            const langToSend = selectedLang === 'auto' ? null : selectedLang;
                            
                            ws.send(JSON.stringify({
                                type: 'audio',
                                data: base64Audio,
                                format: 'webm',
                                language: langToSend
                            }));
                            
                            addMessage('system', '⏳ Processing your message...');
                        };
                        reader.readAsDataURL(audioBlob);
                        
                        // Stop all tracks
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    mediaRecorder.start();
                    document.getElementById('recordingIndicator').classList.add('active');
                    document.getElementById('recordBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                } catch (err) {
                    addMessage('error', '❌ Microphone access denied: ' + err.message);
                    console.error('Microphone error:', err);
                }
            }
            
            function stopRecording() {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                }
            }
            
            function handleMessage(data) {
                switch(data.type) {
                    case 'connected':
                        addMessage('system', '✓ ' + data.message);
                        break;
                    case 'transcription':
                        addMessage('user', data.text);
                        break;
                    case 'response':
                        if (data.crisis) {
                            addMessage('error', '⚠️ CRISIS ALERT: ' + data.text);
                        }
                        addMessage('assistant', data.text);
                        break;
                    case 'tts_request':
                        // Speak the response using browser TTS
                        speakText(data.text, data.language);
                        break;
                    case 'error':
                        addMessage('error', '❌ ' + data.message);
                        break;
                }
            }
            
            function speakText(text, language) {
                // Cancel any ongoing speech first
                window.speechSynthesis.cancel();
                
                // Wait a bit before speaking to ensure cancellation
                setTimeout(() => {
                    const utterance = new SpeechSynthesisUtterance(text);
                    
                    // Set language - improved Hindi support
                    if (language === 'hi') {
                        utterance.lang = 'hi-IN';
                        // Try to find Hindi voice
                        const voices = window.speechSynthesis.getVoices();
                        const hindiVoice = voices.find(voice => voice.lang.startsWith('hi'));
                        if (hindiVoice) {
                            utterance.voice = hindiVoice;
                        }
                    } else {
                        utterance.lang = 'en-US';
                    }
                    
                    utterance.rate = 0.95;
                    utterance.pitch = 1.0;
                    utterance.volume = 1.0;
                    
                    utterance.onerror = (event) => {
                        console.error('Speech synthesis error:', event);
                    };
                    
                    // Speak
                    window.speechSynthesis.speak(utterance);
                }, 100);
            }
            
            // Load voices when they become available
            if (speechSynthesis.onvoiceschanged !== undefined) {
                speechSynthesis.onvoiceschanged = () => {
                    const voices = speechSynthesis.getVoices();
                    console.log('Available voices:', voices.map(v => v.lang + ': ' + v.name));
                };
            }
            
            function addMessage(type, text) {
                const div = document.createElement('div');
                div.className = 'message ' + type;
                div.textContent = text;
                document.getElementById('messages').appendChild(div);
                
                // Smooth scroll to bottom
                const messagesDiv = document.getElementById('messages');
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            // Check microphone support on page load
            window.addEventListener('DOMContentLoaded', () => {
                const protocol = window.location.protocol;
                const hostname = window.location.hostname;
                
                // Show helpful message for HTTP access
                if (protocol === 'http:' && hostname !== 'localhost' && hostname !== '127.0.0.1') {
                    addMessage('system', '🔧 To enable microphone over HTTP, use Chrome with flag:');
                    addMessage('system', '1. Open: chrome://flags/#unsafely-treat-insecure-origin-as-secure');
                    addMessage('system', `2. Add: http://${hostname}:${window.location.port || '8001'}`);
                    addMessage('system', '3. Set to "Enabled" and restart Chrome');
                    addMessage('system', '━'.repeat(50));
                }
                
                // Check if microphone API is available
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    addMessage('error', '⚠️ Microphone API not available in this browser. Please use Chrome or Firefox.');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio processing"""
    await websocket.accept()
    print("[websocket] Client connected")
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Arnish - Your Professional Mental Health AI Assistant"
        })
        
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                try:
                    # Decode base64 audio
                    audio_base64 = data.get("data")
                    audio_format = data.get("format", "webm")
                    forced_language = data.get("language", None)  # Get language hint from client
                    
                    # Convert base64 to bytes
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_in:
                        tmp_in.write(audio_bytes)
                        tmp_in_path = tmp_in.name
                    
                    # Convert to WAV using ffmpeg, then load with librosa
                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    
                    # Use ffmpeg for reliable format conversion
                    import subprocess
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-i', tmp_in_path,
                        '-ar', str(SAMPLE_RATE),  # Resample to 16kHz
                        '-ac', '1',  # Convert to mono
                        '-f', 'wav',
                        tmp_wav
                    ]
                    
                    # Run ffmpeg with suppressed output
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    # Load the converted WAV file
                    audio_data, sr = librosa.load(tmp_wav, sr=SAMPLE_RATE)
                    
                    # Clean up temp files
                    try:
                        os.remove(tmp_in_path)
                        os.remove(tmp_wav)
                    except:
                        pass
                    
                    # Transcribe with forced language if provided
                    text, language = await transcribe_audio(audio_data, language=forced_language)
                    
                    if not text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No speech detected"
                        })
                        continue
                    
                    # Send transcription with detected language
                    await websocket.send_json({
                        "type": "transcription",
                        "text": text,
                        "language": language
                    })
                    
                    print(f"[websocket] Detected language: {language}")
                    
                    # Check for crisis
                    if detect_crisis_keywords(text):
                        crisis_response = "I'm concerned. Please contact emergency services or a crisis helpline." if language == "en" else "मैं चिंतित हूं। कृपया आपातकालीन सेवाओं या संकट हेल्पलाइन से संपर्क करें।"
                        await websocket.send_json({
                            "type": "response",
                            "text": crisis_response,
                            "crisis": True,
                            "language": language
                        })
                        # Still get AI response even in crisis
                    
                    # Get AI response with error handling
                    try:
                        ai_response = await get_ai_response(text, language)
                        print(f"[websocket] AI Response (first 100 chars): {ai_response[:100]}")
                    except Exception as e:
                        print(f"[websocket] AI response failed: {e}")
                        ai_response = "I'm having connection issues. Please try again." if language == "en" else "मुझे कनेक्शन में समस्या है। कृपया फिर से कोशिश करें।"
                    
                    await websocket.send_json({
                        "type": "response",
                        "text": ai_response,
                        "language": language
                    })
                    
                    # Send TTS request (client will call Cloudflare Worker)
                    await websocket.send_json({
                        "type": "tts_request",
                        "text": ai_response,
                        "language": language
                    })
                    
                except Exception as e:
                    print(f"[websocket] Processing error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        print("[websocket] Client disconnected")
    except Exception as e:
        print(f"[websocket] Error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_initialized": genai_client is not None
    }


@app.get("/tts")
async def text_to_speech(text: str, language: str = "en"):
    """Generate speech from text using gTTS"""
    try:
        # Map language codes for gTTS
        lang_map = {
            "en": "en",
            "hi": "hi",
            "hi-IN": "hi"
        }
        tts_lang = lang_map.get(language, "en")
        
        # Create gTTS object
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # Generate audio in memory
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Return the audio file
        return StreamingResponse(audio_buffer, media_type="audio/mpeg")
    except Exception as e:
        print(f"[tts] Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Check for SSL certificate arguments
    ssl_keyfile = None
    ssl_certfile = None
    
    # Parse command line arguments for SSL
    if "--ssl-keyfile" in sys.argv:
        idx = sys.argv.index("--ssl-keyfile")
        ssl_keyfile = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    
    if "--ssl-certfile" in sys.argv:
        idx = sys.argv.index("--ssl-certfile")
        ssl_certfile = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    
    # Run with or without SSL
    if ssl_keyfile and ssl_certfile:
        print(f"🔒 Starting server with SSL/HTTPS")
        print(f"   Key file: {ssl_keyfile}")
        print(f"   Cert file: {ssl_certfile}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile
        )
    else:
        print("⚠️  Starting server without SSL (HTTP only)")
        print("   For HTTPS, run with: --ssl-keyfile <key.pem> --ssl-certfile <cert.pem>")
        uvicorn.run(app, host="0.0.0.0", port=8001)

