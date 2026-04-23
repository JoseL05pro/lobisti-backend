"""
Lobisti Voice Backend - Python (FastAPI)
=========================================
Usa Edge TTS (voces de Microsoft) para voz natural.
"""

import json
import re
import os
import random
import tempfile
import io
import asyncio
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
import edge_tts
import uvicorn

app = FastAPI(title="Lobisti Voice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Respuesta-Texto", "Intent-Name", "Confidence"],
)

# ════════════════════════════════════════════════════════════════
#  CONFIGURACION DE VOZ
# ════════════════════════════════════════════════════════════════

VOZ = "es-MX-JorgeNeural" 
VELOCIDAD = "+15%"
TONO = "+0Hz"

# ════════════════════════════════════════════════════════════════
#  INTENTS
# ════════════════════════════════════════════════════════════════

INTENTS = []
WELCOME_RESPONSE = ""
FALLBACK_RESPONSE = ""
AUDIO_CACHE = {}


def load_intents(path="intents.json"):
    global INTENTS, WELCOME_RESPONSE, FALLBACK_RESPONSE

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for intent in data["intents"]:
        if intent["category"] == "welcome_event":
            WELCOME_RESPONSE = intent["content"][0]["message"]
            continue

        if "fallback" in intent["name"].lower():
            FALLBACK_RESPONSE = intent["content"][0]["message"]
            continue

        if intent.get("trainingPhrases"):
            INTENTS.append({
                "name": intent["name"],
                "phrases": [p.lower().strip() for p in intent["trainingPhrases"]],
                "responses": [
                    c["message"] for c in intent["content"]
                    if c.get("message")
                ],
            })

    print(f"[Lobisti] {len(INTENTS)} intents cargados.")
    print(f"[Lobisti] Welcome: {'OK' if WELCOME_RESPONSE else 'NO'}")
    print(f"[Lobisti] Fallback: {'OK' if FALLBACK_RESPONSE else 'NO'}")


def normalize(text: str) -> str:
    text = text.lower().strip()
    for old, new in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}.items():
        text = text.replace(old, new)
    return re.sub(r"[^\w\s]", "", text)


def tokenize(text: str) -> set:
    stop_words = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "en", "y", "o", "que", "es",
        "por", "con", "para", "se", "su", "me", "te", "lo",
        "le", "nos", "les", "mi", "tu", "como", "mas", "pero",
        "si", "no", "ya", "hay", "ser", "muy", "tambien",
    }
    return set(normalize(text).split()) - stop_words


def match_intent(user_message: str) -> dict:
    user_tokens = tokenize(user_message)

    if not user_tokens:
        return {"intent": "Fallback", "response": FALLBACK_RESPONSE, "confidence": 0.0}

    best_score = 0.0
    best_intent = None

    for intent in INTENTS:
        for phrase in intent["phrases"]:
            phrase_tokens = tokenize(phrase)
            if not phrase_tokens:
                continue
            intersection = user_tokens & phrase_tokens
            union = user_tokens | phrase_tokens
            score = len(intersection) / len(union) if union else 0
            if score > best_score:
                best_score = score
                best_intent = intent

    THRESHOLD = 0.25
    if best_intent and best_score >= THRESHOLD:
        response = random.choice(best_intent["responses"]) if best_intent["responses"] else FALLBACK_RESPONSE
        return {"intent": best_intent["name"], "response": response, "confidence": round(best_score, 3)}

    return {"intent": "Fallback", "response": FALLBACK_RESPONSE, "confidence": round(best_score, 3)}


# ════════════════════════════════════════════════════════════════
#  FUNCIONES DE VOZ
# ════════════════════════════════════════════════════════════════

def strip_emojis(text: str) -> str:
    """Quita TODOS los caracteres no-ASCII para evitar errores con Edge TTS."""
    limpio = re.sub(r'[^\w\s.,;:!?¿¡()\-áéíóúüñÁÉÍÓÚÜÑ]', '', text)
    limpio = re.sub(r'\s+', ' ', limpio).strip()
    return limpio


def transcribir_audio(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 0.5

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        texto = recognizer.recognize_google(audio_data, language="es-MX")
        print(f"[Lobisti] Transcripcion: '{texto}'")
        return texto

    except sr.UnknownValueError:
        print("[Lobisti] No se pudo entender el audio")
        return ""
    except sr.RequestError as e:
        print(f"[Lobisti] Error en Speech Recognition: {e}")
        return ""
    finally:
        os.unlink(tmp_path)


async def generar_audio_respuesta(texto: str) -> bytes:
    """Convierte texto a audio MP3 con Edge TTS (voces naturales de Microsoft)."""
    texto_limpio = strip_emojis(texto)

    if not texto_limpio:
        texto_limpio = "No tengo una respuesta para eso."

    # Revisar cache
    if texto_limpio in AUDIO_CACHE:
        print("[Lobisti] Audio desde cache")
        return AUDIO_CACHE[texto_limpio]

    # Generar con Edge TTS
    try:
        communicate = edge_tts.Communicate(
            text=texto_limpio,
            voice=VOZ,
            rate=VELOCIDAD,
            pitch=TONO,
        )

        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        buffer.seek(0)
        audio_final = buffer.read()

        if not audio_final:
            raise Exception("Edge TTS no genero audio")

    except Exception as e:
        print(f"[Lobisti] Error Edge TTS: {e}, usando gTTS como fallback")
        from gtts import gTTS
        tts = gTTS(text=texto_limpio, lang="es", tld="com.mx", slow=False)
        fallback_buffer = io.BytesIO()
        tts.write_to_fp(fallback_buffer)
        fallback_buffer.seek(0)
        audio_final = fallback_buffer.read()

    print(f"[Lobisti] Audio generado con {VOZ} ({len(audio_final)} bytes)")

    # Guardar en cache (max 100 entradas)
    if len(AUDIO_CACHE) < 100:
        AUDIO_CACHE[texto_limpio] = audio_final

    return audio_final


# ════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.post("/procesar_voz/")
async def procesar_voz(file: UploadFile = File(...)):
    print(f"\n[Lobisti] === Audio recibido: {file.filename} ===")

    audio_bytes = await file.read()
    print(f"[Lobisti] Tamano: {len(audio_bytes)} bytes")

    texto_usuario = transcribir_audio(audio_bytes)

    if not texto_usuario:
        texto_respuesta = FALLBACK_RESPONSE
        intent_name = "Fallback"
        confidence = 0.0
        print("[Lobisti] Audio no reconocido, usando fallback")
    else:
        result = match_intent(texto_usuario)
        texto_respuesta = result["response"]
        intent_name = result["intent"]
        confidence = result["confidence"]
        print(f"[Lobisti] '{texto_usuario}' -> {intent_name} ({confidence})")

    print(f"[Lobisti] Generando voz: {texto_respuesta[:80]}...")

    try:
        audio_respuesta = await generar_audio_respuesta(texto_respuesta)
    except Exception as e:
        print(f"[Lobisti] Error generando audio: {e}")
        from gtts import gTTS
        texto_limpio = strip_emojis(texto_respuesta)
        tts = gTTS(text=texto_limpio or "Error al generar audio", lang="es", tld="com.mx", slow=False)
        fallback_buffer = io.BytesIO()
        tts.write_to_fp(fallback_buffer)
        fallback_buffer.seek(0)
        audio_respuesta = fallback_buffer.read()

    texto_limpio = strip_emojis(texto_respuesta).replace("\n", " ").strip()
    headers = {
        "Respuesta-Texto": quote(texto_limpio, safe=""),
        "Intent-Name": quote(intent_name, safe=""),
        "Confidence": str(confidence),
    }

    return Response(
        content=audio_respuesta,
        media_type="audio/mpeg",
        headers=headers,
    )


@app.post("/chat")
async def chat_texto(data: dict):
    message = data.get("message", "").strip()
    if not message:
        return {"error": "Mensaje vacio"}

    result = match_intent(message)
    print(f"[Lobisti] '{message}' -> {result['intent']} ({result['confidence']})")
    return result


@app.get("/welcome")
async def welcome():
    return {"intent": "Welcome", "response": WELCOME_RESPONSE, "confidence": 1.0}


@app.get("/health")
async def health():
    return {"status": "ok", "intents": len(INTENTS), "voz": VOZ}


# ════════════════════════════════════════════════════════════════
#  CARGAR INTENTS AL IMPORTAR (para Render/uvicorn)
# ════════════════════════════════════════════════════════════════

_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intents.json")
if os.path.exists(_json_path) and not INTENTS:
    load_intents(_json_path)
    print(f"[Lobisti] Voz: {VOZ}")


# ════════════════════════════════════════════════════════════════
#  MAIN (para correr local)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[Lobisti] Servidor de VOZ iniciando en http://localhost:8000")
    print("[Lobisti] Endpoint: POST http://localhost:8000/procesar_voz/")
    print("[Lobisti] Endpoint texto: POST http://localhost:8000/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)