import { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ChatApiClient } from "./chat-api";
import type { Message, ProviderId, ProviderOption, ToolCall } from "./types";
import "./styles.css";

const starters = ["Weather in Chicago", "AAPL stock price", "NBA scores today", "What happened in AI today?"];
const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const chatApi = new ChatApiClient(apiUrl);

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Choose a provider and model, then ask anything. Live questions can use the same MCP tools with either vendor." },
  ]);
  const [providers, setProviders] = useState<ProviderOption[]>([]);
  const [providerId, setProviderId] = useState<ProviderId>("openai");
  const [modelId, setModelId] = useState("");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [error, setError] = useState<string | null>(null);

  const selectedProvider = useMemo(
    () => providers.find(provider => provider.id === providerId),
    [providers, providerId],
  );

  useEffect(() => {
    async function loadModels() {
      try {
        const data = await chatApi.getModels();
        setProviders(data.providers);
        const preferred = data.providers.find(provider => provider.id === data.default_provider && provider.configured)
          ?? data.providers.find(provider => provider.configured)
          ?? data.providers[0];
        if (preferred) {
          setProviderId(preferred.id);
          setModelId(preferred.models[0]?.id ?? "");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to reach the chat API.");
      } finally {
        setModelsLoading(false);
      }
    }
    loadModels();
  }, []);

  function changeProvider(nextProvider: ProviderId) {
    const provider = providers.find(item => item.id === nextProvider);
    setProviderId(nextProvider);
    setModelId(provider?.models[0]?.id ?? "");
    setError(null);
  }

  async function submit(event?: FormEvent, suggested?: string) {
    event?.preventDefault();
    const question = (suggested ?? input).trim();
    if (!question || loading || !modelId) return;
    const nextMessages = [...messages, { role: "user" as const, content: question }];
    const assistantIndex = nextMessages.length;
    setMessages([...nextMessages, {
      role: "assistant",
      content: "",
      provider: providerId,
      model: modelId,
      streaming: true,
    }]);
    setInput("");
    setError(null);
    setToolCalls([]);
    setLoading(true);
    try {
      await chatApi.streamCompletion(
        {
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
          provider: providerId,
          model: modelId,
        },
        (streamEvent, data) => {
          if (streamEvent === "error") throw new Error(data.detail ?? "The streamed response failed.");
          if (streamEvent === "delta" && data.content) {
            setMessages(current => current.map((message, index) => index === assistantIndex
              ? { ...message, content: message.content + data.content }
              : message));
          }
          if (streamEvent === "tool_call" && data.name && data.arguments) {
            setToolCalls(current => [...current, {
              name: data.name!,
              arguments: data.arguments!,
              result: data.result,
            }]);
          }
          if (streamEvent === "done") {
            setMessages(current => current.map((message, index) => index === assistantIndex
              ? {
                  ...message,
                  provider: data.provider ?? message.provider,
                  model: data.model ?? message.model,
                  streaming: false,
                }
              : message));
          }
        },
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reach the chat API.");
      setMessages(current => {
        const assistant = current[assistantIndex];
        if (!assistant) return current;
        if (!assistant.content) return current.filter((_, index) => index !== assistantIndex);
        return current.map((message, index) => index === assistantIndex
          ? { ...message, streaming: false }
          : message);
      });
    } finally {
      setLoading(false);
    }
  }

  return <main>
    <section className="shell" aria-label="MCP Chat Lab">
      <header>
        <div className="mark">M</div>
        <div className="brand"><h1>MCP Chat Lab</h1><p>One tool layer, multiple model vendors</p></div>
        <span className="status"><i /> API ready</span>
      </header>

      <section className="model-bar" aria-label="Model selection">
        <label>
          <span>Provider</span>
          <select
            value={providerId}
            onChange={event => changeProvider(event.target.value as ProviderId)}
            disabled={loading || modelsLoading}
          >
            {providers.map(provider => <option key={provider.id} value={provider.id} disabled={!provider.configured}>
              {provider.name}{provider.configured ? "" : " · API key required"}
            </option>)}
          </select>
        </label>
        <label>
          <span>Model</span>
          <select value={modelId} onChange={event => setModelId(event.target.value)} disabled={loading || modelsLoading}>
            {(selectedProvider?.models ?? []).map(model => <option key={model.id} value={model.id}>{model.name}</option>)}
          </select>
        </label>
        <div className={`provider-state ${selectedProvider?.configured ? "ready" : "missing"}`}>
          <b>{selectedProvider?.configured ? "Ready" : "Not configured"}</b>
          <span>{selectedProvider?.configured ? "Server-side key detected" : "Add the provider key to .env"}</span>
        </div>
      </section>

      <div className="chat" aria-live="polite">
        {messages.map((message, index) => <article className={`message ${message.role}`} key={index}>
          <span className="avatar">{message.role === "assistant" ? "M" : "You"}</span>
          <div className="message-body">
            {message.model && <small>{message.provider === "gemini" ? "Google Gemini" : "OpenAI"} · {message.model}</small>}
            <p className={message.streaming ? "streaming" : undefined}>
              {message.content || "Waiting for response…"}
            </p>
          </div>
        </article>)}
      </div>
      {toolCalls.length > 0 && <details className="tools" open>
        <summary>Live MCP tools used ({toolCalls.length})</summary>
        {toolCalls.map((call, index) => <div className="tool" key={index}><code>{call.name}</code><span>{JSON.stringify(call.arguments)}</span></div>)}
      </details>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="suggestions">
        {starters.map(suggestion => <button key={suggestion} onClick={() => submit(undefined, suggestion)} disabled={loading || !selectedProvider?.configured}>{suggestion}</button>)}
      </div>
      <form onSubmit={submit}>
        <label className="sr-only" htmlFor="prompt">Your message</label>
        <textarea id="prompt" value={input} onChange={event => setInput(event.target.value)} placeholder="Ask anything…" rows={2} disabled={loading || !selectedProvider?.configured} />
        <button className="send" type="submit" disabled={loading || !input.trim() || !selectedProvider?.configured}>Send <span>↗</span></button>
      </form>
      <footer>Keys remain on the API server. Model choices are server-controlled; live facts come from MCP services.</footer>
    </section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
