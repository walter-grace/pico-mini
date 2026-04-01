import http from "node:http";
import { URL } from "node:url";

export interface Message {
  role: "system" | "user" | "assistant";
  content: string | any[];  // string for text, array for multimodal (image_url)
}

export interface CompletionChunk {
  choices: Array<{
    delta: { content?: string };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ServerHealth {
  status: string;
  model?: string;
  slots_idle?: number;
  slots_processing?: number;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (stats: { tokensPerSec: number; totalTokens: number }) => void;
  onError: (error: string) => void;
}

export class LlamaClient {
  private baseUrl: string;

  constructor(serverUrl: string) {
    this.baseUrl = serverUrl.replace(/\/+$/, "");
  }

  async health(): Promise<ServerHealth> {
    return this.get("/health");
  }

  async chat(messages: Message[], maxTokens = 100): Promise<string> {
    const resp = await this.post("/v1/chat/completions", {
      messages,
      temperature: 0,
      max_tokens: maxTokens,
      stream: false,
    });
    const c = resp?.choices?.[0]?.message;
    return c?.content || c?.reasoning_content || "";
  }

  async getModelName(): Promise<string> {
    try {
      const resp = await this.get("/v1/models");
      if (resp?.data?.[0]?.id) return resp.data[0].id;
      return "unknown";
    } catch {
      return "unknown";
    }
  }

  async streamChat(
    messages: Message[],
    callbacks: StreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const url = new URL(this.baseUrl + "/v1/chat/completions");
    const body = JSON.stringify({
      messages,
      stream: true,
      temperature: 0.7,
      max_tokens: 2048,
    });

    return new Promise((resolve, reject) => {
      const req = http.request(
        url,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          signal,
        },
        (res) => {
          if (res.statusCode !== 200) {
            callbacks.onError(`Server returned ${res.statusCode}`);
            reject(new Error(`HTTP ${res.statusCode}`));
            return;
          }

          let buffer = "";
          let totalTokens = 0;
          const startTime = Date.now();
          let done = false;

          res.setEncoding("utf-8");
          res.on("data", (chunk: string) => {
            buffer += chunk;
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || !trimmed.startsWith("data: ")) continue;
              const data = trimmed.slice(6);
              if (data === "[DONE]") {
                if (!done) {
                  done = true;
                  const elapsed = (Date.now() - startTime) / 1000;
                  const tokPerSec =
                    elapsed > 0 ? totalTokens / elapsed : 0;
                  callbacks.onDone({
                    tokensPerSec: Math.round(tokPerSec * 10) / 10,
                    totalTokens,
                  });
                }
                resolve();
                return;
              }
              try {
                const parsed: CompletionChunk = JSON.parse(data);
                const content = parsed.choices?.[0]?.delta?.content;
                if (content) {
                  totalTokens++;
                  callbacks.onToken(content);
                }
                if (parsed.choices?.[0]?.finish_reason === "stop" && !done) {
                  done = true;
                  const elapsed = (Date.now() - startTime) / 1000;
                  const tokPerSec =
                    elapsed > 0 ? totalTokens / elapsed : 0;
                  callbacks.onDone({
                    tokensPerSec: Math.round(tokPerSec * 10) / 10,
                    totalTokens,
                  });
                  resolve();
                  return;
                }
              } catch {
                // skip malformed JSON chunks
              }
            }
          });

          res.on("end", () => {
            if (!done) {
              done = true;
              const elapsed = (Date.now() - startTime) / 1000;
              const tokPerSec = elapsed > 0 ? totalTokens / elapsed : 0;
              callbacks.onDone({
                tokensPerSec: Math.round(tokPerSec * 10) / 10,
                totalTokens,
              });
            }
            resolve();
          });

          res.on("error", (err) => {
            callbacks.onError(err.message);
            reject(err);
          });
        }
      );

      req.on("error", (err) => {
        callbacks.onError(err.message);
        reject(err);
      });

      req.write(body);
      req.end();
    });
  }

  private get(path: string): Promise<any> {
    const url = new URL(this.baseUrl + path);
    return new Promise((resolve, reject) => {
      http.get(url, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve({ raw: data });
          }
        });
        res.on("error", reject);
      }).on("error", reject);
    });
  }

  private post(path: string, body: any): Promise<any> {
    const url = new URL(this.baseUrl + path);
    const data = JSON.stringify(body);
    return new Promise((resolve, reject) => {
      const req = http.request(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }, (res) => {
        let buf = "";
        res.on("data", (chunk) => (buf += chunk));
        res.on("end", () => {
          try { resolve(JSON.parse(buf)); }
          catch { resolve({ raw: buf }); }
        });
        res.on("error", reject);
      });
      req.on("error", reject);
      req.write(data);
      req.end();
    });
  }
}
