from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import requests
from rag_tfidf import TfidfRAG
import re

# Add the server directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_endpoints.courses.handler import CoursesHandler, CourseDetailHandler
from api_endpoints.students.handler import StudentProgressHandler
from api_endpoints.predictions.handler import CreatePredictionHandler

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "docs")
rag = TfidfRAG(DOCS_DIR)
rag.build()


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "AI Academy API"})


# Course endpoints
@app.route("/api/courses", methods=["GET"])
def get_courses():
    return CoursesHandler(request)


@app.route("/api/courses/<int:course_id>", methods=["GET"])
def get_course(course_id):
    return CourseDetailHandler(request, course_id)


@app.route("/api/courses/<int:course_id>/enroll", methods=["POST"])
def enroll_in_course(course_id):
    return jsonify({"message": "Enrollment functionality coming soon"})


# Student endpoints
@app.route("/api/students/<int:student_id>/progress", methods=["GET"])
def get_student_progress(student_id):
    return StudentProgressHandler(request, student_id)


# AI/ML prediction endpoint
@app.route("/api/predict", methods=["POST"])
def predict():
    return CreatePredictionHandler(request)


# Function for Query Rewriting, Intent Detection, and Retrieval Confidence Assessment:
def normalize_question(q: str) -> str:
    q0 = q.strip()
    ql = q0.lower()

    mapping = {
        "location": "What locations are mentioned in the provided materials?",
        "locations": "What locations are mentioned in the provided materials?",
        "mission": "What is the mission described in the provided materials?",
        "contact": "What contact information is mentioned in the provided materials?",
        "phone": "What phone number is mentioned in the provided materials?",
        "services": "What services are mentioned in the provided materials?",
    }
    return mapping.get(ql, q0)


def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["where", "location", "address"]):
        return "location"
    if any(k in ql for k in ["phone", "email", "contact"]):
        return "contact"
    if any(k in ql for k in ["list", "which", "what are"]):
        return "list"
    return "general"


def should_ask_clarification(question: str, hits):
    # If no hits were found, ask for clarification
    if not hits:
        return True, "I don't know based on the provided materials."

    top_score = hits[0][1]
    second_score = hits[1][1] if len(hits) > 1 else 0.0

    if top_score < 0.12:
        return True, "I don't know based on the provided materials. Please ask a more specific question."

    if top_score < 0.18 and abs(top_score - second_score) < 0.03:
        return True, "Your question seems ambiguous in the provided materials. Please be more specific."

    return False, None


# -----------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")  # You can also change to llama2


# Flow: handling the chat
@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True) or {}
    question = (payload.get("message") or "").strip()

    if not question:
        return jsonify({"error": "message is required"}), 400

    top_k = int(payload.get("top_k") or 3)

    # Rewriting the question for better accuracy
    search_query = normalize_question(question)
    hits = rag.query(search_query, top_k=top_k)

    # If the retrieval is unclear, ask for clarification
    need_clarify, msg = should_ask_clarification(question, hits)
    if need_clarify:
        return jsonify({"answer": msg, "citations": []})

    # Construct the context for LLM
    # 构建上下文和引用信息（这里我们会标明文件名以及具体来源）
    context_blocks = []
    citations = []
    for rank, (chunk, score) in enumerate(hits, start=1):
        # 添加源文件和内容
        context_blocks.append(
            f"[{rank}] SOURCE: {chunk.source} (chunk {chunk.chunk_id})\n{chunk.text}"
        )
        citations.append({
            "rank": rank,
            "source": chunk.source,  # 文件名（例如：location.txt）
            "chunk_id": chunk.chunk_id,
            "score": round(score, 4),
            "excerpt": chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
        })

    context = "\n\n---\n\n".join(context_blocks)

    # Determine output format based on the question intent
    intent = detect_intent(question)
    format_instruction = {
        "location": "Return a short bullet list of locations only.",
        "contact": "Return only the contact details found.",
        "list": "Return a concise bullet list.",
        "general": "Return 1-3 concise sentences."
    }[intent]

    # Define system prompt for LLM generation
    system_prompt = (
        "You are a concise RAG QA assistant.\n"
        "Answer ONLY using the provided CONTEXT.\n"
        "Do not use outside knowledge.\n"
        "If the answer is not explicitly supported by the CONTEXT, say: "
        "\"I don't know based on the provided materials.\"\n"
        "Do not repeat the source text verbatim unless necessary.\n"
        "Merge overlapping or repeated evidence into one natural answer.\n"
        "If the user asks for one fact, return only that fact.\n"
        f"{format_instruction}\n"
        "At the end include: Sources: [numbers]\n\n"
        f"CONTEXT:\n{context}"
    )

    # Request LLM for an answer
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 160},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("message", {}).get("content", "")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer, "citations": citations})


#------------------------------
# Serve static files (for production deployment)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
