import React, { useState } from "react";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:5000";

export default function Chatbox() {
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function ask() {
    const q = message.trim();
    if (!q) return;

    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      const data = await res.json();
      setAnswer(data.answer || "");
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "40px auto", padding: 16 }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 12 }}>
        Catholic Health Q&amp;A (Demo)
      </h1>

      <p style={{ marginBottom: 16 }}>
        Ask a question about the company. The backend answers using the provided
        materials.
      </p>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") ask();
          }}
          placeholder="e.g., What is Catholic Health's mission?"
          style={{
            flex: 1,
            padding: 12,
            borderRadius: 8,
            border: "1px solid #ccc",
          }}
        />
        <button
          onClick={ask}
          disabled={loading}
          style={{
            padding: "12px 16px",
            borderRadius: 8,
            border: "1px solid #ccc",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Asking..." : "Ask"}
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 16, color: "crimson" }}>
          Error: {error}
        </div>
      )}

      {answer && (
        <div
          style={{
            marginTop: 16,
            padding: 16,
            border: "1px solid #eee",
            borderRadius: 12,
            whiteSpace: "pre-wrap",
            lineHeight: 1.5,
          }}
        >
          {answer}
        </div>
      )}
    </div>
  );
}
