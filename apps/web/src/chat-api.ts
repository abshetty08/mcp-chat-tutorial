import { ServerSentEventStream } from "./event-stream";
import type { Message, ModelsResponse, ProviderId, StreamEventHandler } from "./types";

type StreamChatRequest = {
  messages: Pick<Message, "role" | "content">[];
  provider: ProviderId;
  model: string;
};

export class ChatApiClient {
  constructor(private readonly baseUrl: string) {}

  async getModels(): Promise<ModelsResponse> {
    const response = await fetch(`${this.baseUrl}/v1/models`);
    if (!response.ok) throw new Error("Unable to load the model catalog.");
    return response.json() as Promise<ModelsResponse>;
  }

  async streamCompletion(request: StreamChatRequest, onEvent: StreamEventHandler): Promise<void> {
    const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...request, stream: true }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail ?? "The chat API returned an error.");
    }
    await new ServerSentEventStream(onEvent).consume(response);
  }
}
