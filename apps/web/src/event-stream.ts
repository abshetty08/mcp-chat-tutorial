import type { StreamEventHandler, StreamPayload } from "./types";

export class ServerSentEventStream {
  private readonly decoder = new TextDecoder();
  private buffer = "";

  constructor(private readonly onEvent: StreamEventHandler) {}

  async consume(response: Response): Promise<void> {
    if (!response.body) throw new Error("This browser did not provide a streaming response body.");
    const reader = response.body.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        this.buffer += this.decoder.decode(value, { stream: !done });
        this.dispatchCompleteFrames();
        if (done) break;
      }
      if (this.buffer.trim()) this.dispatch(this.buffer);
    } finally {
      reader.releaseLock();
    }
  }

  private dispatchCompleteFrames(): void {
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary >= 0) {
      this.dispatch(this.buffer.slice(0, boundary));
      this.buffer = this.buffer.slice(boundary + 2);
      boundary = this.buffer.indexOf("\n\n");
    }
  }

  private dispatch(block: string): void {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (data.length) this.onEvent(event, JSON.parse(data.join("\n")) as StreamPayload);
  }
}
