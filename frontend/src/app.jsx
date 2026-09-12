import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function ConfidenceBadge({ confidence }) {
  if (!confidence) return null;
  const colors = {
    high: "#1e8e3e",
    medium: "#e37400",
    low: "#c53929",
    none: "#c53929",
  };
  return (
    <span
      className="confidence-badge"
      style={{ background: colors[confidence] || "#999" }}
    >
      {confidence} confidence
    </span>
  );
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? "Hide sources" : `Show ${sources.length} source(s)`}
      </button>
      {open && (
        <ul>
          {sources.map((s) => (
            <li key={s.id}>
              <strong>{s.question}</strong>{" "}
              <span className="score">({s.score.toFixed(2)})</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      from: "bot",
      text: "Hello! Ask a medical question. I answer from a medical Q&A knowledge base — this isn't a substitute for professional medical advice.",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim() || loading) return;

    const userMsg = input;
    setMessages((prev) => [...prev, { from: "user", text: userMsg }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg }),
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          from: "bot",
          text: data.answer || "No answer found.",
          confidence: data.confidence,
          sources: data.sources,
          disclaimer: data.disclaimer,
        },
      ]);
    } catch (err) {
      console.log(err);
      setMessages((prev) => [
        ...prev,
        { from: "bot", text: "Error contacting server. Is the backend running?" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-container">
      <h2>MedQA Assistant</h2>
      <p className="app-disclaimer">
        ⚠️ General health information only — not a substitute for professional medical advice.
      </p>

      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.from}`}>
            <div className="bubble">
              {m.text}
              {m.confidence && <ConfidenceBadge confidence={m.confidence} />}
              {m.sources && <Sources sources={m.sources} />}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="bubble">Thinking…</div>
          </div>
        )}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask something..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}
