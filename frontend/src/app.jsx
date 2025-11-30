import { useState } from "react";

export default function App() {
  const [messages, setMessages] = useState([
    { from: "bot", text: "Hello! Ask a medical question." }
  ]);

  const [input, setInput] = useState("");

  async function sendMessage() {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages(prev => [...prev, { from: "user", text: userMsg }]);
    setInput("");

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg })
      });

      const data = await res.json();

      setMessages(prev => [
        ...prev,
        { from: "bot", text: data.answer || "No answer found." }
      ]);
    } catch (err) {
      console.log(err);
      setMessages(prev => [
        ...prev,
        { from: "bot", text: "Error contacting server." }
      ]);
    }
  }

  return (
    <div className="chat-container">
      <h2>MedQA Assistant</h2>

      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.from}`}>
            <div className="bubble">{m.text}</div>
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder="Ask something..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}
