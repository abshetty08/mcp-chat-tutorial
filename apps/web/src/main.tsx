import { FormEvent, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Message = { role: "user" | "assistant"; content: string };
type ToolCall = { name: string; arguments: Record<string, unknown>; result: unknown };

const starters = ["Weather in Chicago", "AAPL stock price", "NBA scores today", "What happened in AI today?"];
const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Ask a question, or try a live-data example below. I can use MCP tools when fresh information matters." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function submit(event?: FormEvent, suggested?: string) {
    event?.preventDefault();
    const question = (suggested ?? input).trim();
    if (!question || loading) return;
    const nextMessages = [...messages, { role: "user" as const, content: question }];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setToolCalls([]);
    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "The chat API returned an error.");
      setMessages([...nextMessages, { role: "assistant", content: data.answer }]);
      setToolCalls(data.tool_calls ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reach the chat API.");
    } finally {
      setLoading(false);
    }
  }

  return <main>
    <section className="shell" aria-label="MCP Chat Lab">
      <header>
        <div className="mark">M</div>
        <div><h1>MCP Chat Lab</h1><p>React client · FastAPI gateway · live MCP tools</p></div>
        <span className="status"><i /> API ready</span>
      </header>
      <div className="chat" aria-live="polite">
        {messages.map((message, index) => <article className={`message ${message.role}`} key={index}>
          <span className="avatar">{message.role === "assistant" ? "M" : "You"}</span>
          <p>{message.content}</p>
        </article>)}
        {loading && <article className="message assistant"><span className="avatar">M</span><p className="typing">Checking sources and composing an answer</p></article>}
      </div>
      {toolCalls.length > 0 && <details className="tools" open>
        <summary>Live MCP tools used ({toolCalls.length})</summary>
        {toolCalls.map((call, index) => <div className="tool" key={index}><code>{call.name}</code><span>{JSON.stringify(call.arguments)}</span></div>)}
      </details>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="suggestions">
        {starters.map(suggestion => <button key={suggestion} onClick={() => submit(undefined, suggestion)} disabled={loading}>{suggestion}</button>)}
      </div>
      <form onSubmit={submit}>
        <label className="sr-only" htmlFor="prompt">Your message</label>
        <textarea id="prompt" value={input} onChange={event => setInput(event.target.value)} placeholder="Ask anything…" rows={2} disabled={loading} />
        <button className="send" type="submit" disabled={loading || !input.trim()}>Send <span>↗</span></button>
      </form>
      <footer>Live information comes from MCP services. Verify consequential information independently.</footer>
    </section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
