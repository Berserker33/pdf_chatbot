from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from flask_cors import CORS
import os, uuid, re, threading
from collections import OrderedDict
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
CORS(app)

# Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Session store
pdf_store = OrderedDict()
MAX_SESSIONS = 50

def store_session(sid, text):
    pdf_store[sid] = text
    pdf_store.move_to_end(sid)
    if len(pdf_store) > MAX_SESSIONS:
        pdf_store.popitem(last=False)

def extract_text(path):
    import pdfplumber
    text = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
    return text

def delete_async(path):
    def f():
        try: os.remove(path)
        except: pass
    threading.Thread(target=f).start()

# STREAM
def ask_stream(text, question):
    prompt = f"""
Answer ONLY from document.
If not found say: Not in document.

DOCUMENT:
{text[:12000]}

QUESTION:
{question}
"""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    ):
        if chunk.text:
            yield chunk.text

# ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["pdf"]
    sid = str(uuid.uuid4())
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f.filename)
    path = os.path.join(UPLOAD_FOLDER, sid + "_" + name)

    f.save(path)
    text = extract_text(path)
    delete_async(path)

    store_session(sid, text)

    return jsonify({
        "session_id": sid,
        "stats": {
            "characters": len(text),
            "words": len(text.split()),
            "truncated": len(text) > 12000
        }
    })

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json
    sid = data["session_id"]
    q = data["question"]

    def generate():
        for t in ask_stream(pdf_store[sid], q):
            yield t

    return Response(stream_with_context(generate()), mimetype="text/plain")

# RUN
if __name__ == "__main__":
    app.run(debug=True)