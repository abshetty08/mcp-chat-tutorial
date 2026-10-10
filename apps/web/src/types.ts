export type ProviderId = "openai" | "gemini";

export type Message = {
  role: "user" | "assistant";
  content: string;
  provider?: ProviderId;
  model?: string;
  streaming?: boolean;
};

export type ToolCall = {
  name: string;
  arguments: Record<string, unknown>;
  result: unknown;
};

export type ModelOption = { id: string; name: string };

export type ProviderOption = {
  id: ProviderId;
  name: string;
  configured: boolean;
  models: ModelOption[];
};

export type ModelsResponse = {
  providers: ProviderOption[];
  default_provider: ProviderId;
};

export type StreamPayload = {
  content?: string;
  detail?: string;
  provider?: ProviderId;
  model?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  result?: unknown;
};

export type StreamEventHandler = (event: string, payload: StreamPayload) => void;
