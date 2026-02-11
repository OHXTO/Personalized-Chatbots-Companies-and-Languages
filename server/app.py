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
    # TODO: Implement enrollment logic
    return jsonify({"message": "Enrollment functionality coming soon"})

# Student endpoints
@app.route("/api/students/<int:student_id>/progress", methods=["GET"])
def get_student_progress(student_id):
    return StudentProgressHandler(request, student_id)

# AI/ML prediction endpoint
@app.route("/api/predict", methods=["POST"])
def predict():
    return CreatePredictionHandler(request)


# -----------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")  # 你也可以改 llama2


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "message is required"}), 400


    tokens = re.findall(r"[a-z0-9]+", question.lower())
    if len(tokens) <= 1:
        return jsonify({
            "answer": 'Your question is a bit ambiguous. Did you mean "locations" (hospital sites), or something else (e.g., local services, local contact, local clinics)? Please specify.',
            "citations": []
        })


    def guess_source_hint(question: str):
        # 取问题里的关键词（字母数字），做成候选
        tokens = re.findall(r"[a-z0-9]+", question.lower())
        # 常见停用词过滤，避免选到无意义的词
        stop = {"what","is","the","a","an","of","to","for","in","on","and","or","about","tell","me","please"}
        tokens = [t for t in tokens if t not in stop]
        # 取最可能的前几个（短问题很少）
        return tokens[:3]  # 例如 "mission" / "location" / "faq"

    # 可选：传 top_k（单点问题建议 2 更准）
    top_k = int(payload.get("top_k") or 2)


    # 先用文件名 hint 做过滤检索；如果找不到再退回全量
    hints = guess_source_hint(question)

    hits = []
    for h in hints:
        hits = rag.query(question, top_k=top_k, source_contains=h)
        if hits:
            break
    
    # 如果最相关的相似度都很低，说明检索不靠谱 → 追问澄清
    if hits and hits[0][1] < 0.08:
        return jsonify({
            "answer": 'I’m not sure which topic you mean from the provided materials. Can you rephrase or be more specific (e.g., "hospital locations", "mission", "MyChart", "contact number")?',
            "citations": []
        })

    if not hits:
        hits = rag.query(question, top_k=top_k)


    # 组装只包含相关段落的上下文
    context_blocks = []
    citations = []
    for rank, (chunk, score) in enumerate(hits, start=1):
        context_blocks.append(
            f"[{rank}] SOURCE: {chunk.source} (chunk {chunk.chunk_id})\n{chunk.text}"
        )
        citations.append({
            "rank": rank,
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "score": round(score, 4),
            "excerpt": chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
        })

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a concise QA assistant for Catholic Health - Long Island.\n"
        "Answer ONLY the user's question using ONLY the provided CONTEXT.\n"
        "Do NOT add extra facts (mission/vision/objectives etc.) unless the user asked.\n"
        "If the question is asking for ONE attribute (e.g., locations), answer ONLY that.\n"
        "If the answer is not explicitly in the CONTEXT, say: \"I don't know based on the provided materials.\" \n"
        "Only include information that directly answers the question.\n"
        "Format:\n"
        "- Answer: <one short paragraph or up to 3 bullets>\n"
        "- Sources: [numbers]\n\n"
        f"CONTEXT:\n{context}"
    )

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 220},
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