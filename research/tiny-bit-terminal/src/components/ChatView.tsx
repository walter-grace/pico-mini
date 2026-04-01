import React, { useState, useCallback, useRef, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { LlamaClient, Message } from "../api.js";
import { RetroSpinner } from "./RetroSpinner.js";
import { execSync } from "node:child_process";

interface ChatEntry {
  role: "user" | "assistant" | "system" | "info" | "error";
  content: string;
}

// Compare servers — Bonsai on 8203, Qwen3.5-9B on 8204
const COMPARE_SERVERS = [
  { name: "Bonsai-8B (1-bit)", url: "http://localhost:8203" },
  { name: "Qwen3.5-9B (IQ2)", url: "http://localhost:8204" },
];

interface ChatViewProps {
  client: LlamaClient;
  onQuit: () => void;
  onStatsUpdate: (stats: { tokPerSec: number; totalTokens: number }) => void;
  onStatusChange: (status: "connected" | "disconnected" | "streaming") => void;
}

const HELP_TEXT = `
╔══════════════════════════════════════════╗
║   tiny bit Commands                     ║
╠══════════════════════════════════════════╣
║ Just type to chat with the model        ║
║                                          ║
║ /search <query>  Web search (DuckDuckGo) ║
║ /image <path>    Describe an image       ║
║ /screenshot      Capture & describe      ║
║ /shell <cmd>     Run a shell command     ║
║ /stats           Show server stats       ║
║ /compare <prompt> Compare Bonsai vs 9B    ║
║ /clear           Clear chat history      ║
║ /help            Show this help          ║
║ /quit            Exit tiny bit           ║
╚══════════════════════════════════════════╝`;

export const ChatView: React.FC<ChatViewProps> = ({
  client,
  onQuit,
  onStatsUpdate,
  onStatusChange,
}) => {
  const [entries, setEntries] = useState<ChatEntry[]>([
    {
      role: "system",
      content:
        '  Welcome to tiny bit. Type a message or /help for commands.',
    },
  ]);
  const [input, setInput] = useState("");
  const [showCommands, setShowCommands] = useState(false);
  const [spinnerMsg, setSpinnerMsg] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [scrollOffset, setScrollOffset] = useState(0);
  const messagesRef = useRef<Message[]>([
    {
      role: "system",
      content:
        `You are tiny bit, a helpful AI assistant running locally on Apple Silicon. Today is ${new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}. You respond concisely and helpfully.

You have access to these tools. To use a tool, include a tool_call tag in your response:

<tools>
[
  {"name": "shell", "description": "Run a shell command on macOS and see the output", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to execute"}}, "required": ["command"]}},
  {"name": "web_search", "description": "Search the web using DuckDuckGo", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}},
  {"name": "read_file", "description": "Read a text file or analyze an image/PDF. Supports .txt, .md, .py, .js, .json, .csv, .png, .jpg, .pdf and more.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path to read or analyze"}}, "required": ["path"]}}
]
</tools>

To call a tool, output EXACTLY this format (no other text before it):
<tool_call>{"name": "shell", "arguments": {"command": "ls ~/Desktop"}}</tool_call>

After the tool runs, you will receive the result and can then respond to the user.
Rules:
- When the user asks about files, folders, or their system, USE the shell tool directly. Do not just suggest commands.
- When the user asks about current events or news, USE the web_search tool.
- You can call multiple tools in sequence.
- After seeing tool results, give a clear summary to the user.`,
    },
  ]);
  const abortRef = useRef<AbortController | null>(null);

  // Visible entries for display (last N entries)
  const MAX_VISIBLE = 16;
  const visibleEntries = entries.slice(
    Math.max(0, entries.length - MAX_VISIBLE - scrollOffset),
    entries.length - scrollOffset > 0 ? entries.length - scrollOffset : undefined
  );

  useInput((ch, key) => {
    if (key.escape && isStreaming) {
      abortRef.current?.abort();
      setIsStreaming(false);
      setEntries((prev) => [
        ...prev,
        { role: "info", content: "◆ Generation stopped." },
      ]);
      onStatusChange("connected");
    }
    // Scroll with ctrl+up/down
    if (key.upArrow && key.ctrl) {
      setScrollOffset((prev) => Math.min(prev + 1, Math.max(0, entries.length - MAX_VISIBLE)));
    }
    if (key.downArrow && key.ctrl) {
      setScrollOffset((prev) => Math.max(prev - 1, 0));
    }
  });

  const handleSubmit = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;

      setInput("");
      setScrollOffset(0);

      // Handle commands
      if (trimmed.startsWith("/")) {
        const parts = trimmed.split(/\s+/);
        const cmd = parts[0].toLowerCase();
        const args = parts.slice(1).join(" ");

        switch (cmd) {
          case "/quit":
          case "/exit":
            onQuit();
            return;

          case "/help":
            setEntries((prev) => [
              ...prev,
              { role: "info", content: HELP_TEXT },
            ]);
            return;

          case "/clear":
            setEntries([
              {
                role: "system",
                content: "  Chat cleared.",
              },
            ]);
            messagesRef.current = [messagesRef.current[0]];
            return;

          case "/compare": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /compare <prompt>" },
              ]);
              return;
            }
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `/compare ${args}` },
              { role: "info", content: "◆ Racing both models simultaneously..." },
            ]);

            const compareMessages: Message[] = [
              { role: "system", content: "Answer concisely in 2-3 sentences." },
              { role: "user", content: args },
            ];

            // Track both streams simultaneously
            const results: { [key: string]: string } = {};
            const stats: { [key: string]: { tokPerSec: number; tokens: number } } = {};
            let activeStreams = 0;

            setIsStreaming(true);
            onStatusChange("streaming");

            const updateStreamDisplay = () => {
              const lines: string[] = [];
              for (const server of COMPARE_SERVERS) {
                const text = results[server.name] || "";
                const s = stats[server.name];
                lines.push(`┌─ ${server.name} ${s ? `(${s.tokPerSec} tok/s)` : "(streaming...)"}`);
                lines.push(`│ ${text || "..."}`);
                lines.push(`└${"─".repeat(40)}`);
              }
              setStreamingText(lines.join("\n"));
            };

            // Fire both requests at the same time
            const promises = COMPARE_SERVERS.map(async (server) => {
              try {
                const compareClient = new LlamaClient(server.url);
                results[server.name] = "";
                activeStreams++;

                await new Promise<void>((resolve) => {
                  compareClient.streamChat(
                    compareMessages,
                    {
                      onToken: (token) => {
                        results[server.name] += token;
                        updateStreamDisplay();
                      },
                      onDone: (s) => {
                        stats[server.name] = { tokPerSec: s.tokensPerSec, tokens: s.totalTokens };
                        activeStreams--;
                        updateStreamDisplay();
                        resolve();
                      },
                      onError: (err) => {
                        results[server.name] = `ERROR: ${err}`;
                        activeStreams--;
                        updateStreamDisplay();
                        resolve();
                      },
                    }
                  );
                });
              } catch (e: any) {
                results[server.name] = `ERROR: ${e.message?.split("\n")[0]}`;
              }
            });

            // Wait for BOTH to finish
            await Promise.all(promises);

            // Add final results to chat history
            for (const server of COMPARE_SERVERS) {
              const s = stats[server.name];
              setEntries((prev) => [
                ...prev,
                { role: "info", content: `◆ ─── ${server.name} ${s ? `· ${s.tokPerSec} tok/s · ${s.tokens} tokens` : ""} ───` },
                { role: "assistant", content: results[server.name] || "No response" },
              ]);
            }

            setIsStreaming(false);
            setStreamingText("");
            onStatusChange("connected");
            return;
          }

          case "/stats": {
            try {
              const health = await client.health();
              setEntries((prev) => [
                ...prev,
                {
                  role: "info",
                  content: `◆ Server Status: ${health.status}\n  Idle slots: ${health.slots_idle ?? "?"}\n  Processing: ${health.slots_processing ?? "?"}`,
                },
              ]);
            } catch (e: any) {
              setEntries((prev) => [
                ...prev,
                {
                  role: "error",
                  content: `◆ Error: ${e.message}`,
                },
              ]);
            }
            return;
          }

          case "/shell": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /shell <task or command>" },
              ]);
              return;
            }
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `/shell ${args}` },
            ]);

            // Check if it looks like a real command or natural language
            const looksLikeCommand = /^(ls|cd|cat|pwd|echo|grep|find|df|du|ps|top|whoami|date|uname|which|file|head|tail|wc|sort|mkdir|rm|cp|mv|chmod|chown|curl|wget|python|node|npm|git|brew|pip)\b/.test(args.trim());

            let cmd = args;
            if (!looksLikeCommand) {
              // Ask LLM to generate command
              setSpinnerMsg("Generating command...");
              try {
                const genResp = await client.chat([
                  { role: "system", content: "Generate a single macOS shell command for the user's request. Output ONLY the command, nothing else." },
                  { role: "user", content: args },
                ]);
                cmd = genResp.trim().replace(/^`+|`+$/g, "");
                setSpinnerMsg(`Executing: $ ${cmd}`);
                setEntries((prev) => [...prev, { role: "info", content: `◆ Running: $ ${cmd}` }]);
              } catch {
                setEntries((prev) => [...prev, { role: "error", content: "◆ Failed to generate command" }]);
                return;
              }
            }

            try {
              const output = execSync(cmd, {
                encoding: "utf-8",
                timeout: 10000,
                maxBuffer: 1024 * 1024,
              }).trim();
              setSpinnerMsg("");
              const truncOutput = output.split("\n").slice(0, 20).join("\n");
              setEntries((prev) => [
                ...prev,
                { role: "info", content: `◆ $ ${cmd}\n${truncOutput}` },
              ]);
              // Feed output to LLM for synthesis
              setSpinnerMsg("Summarizing...");
              messagesRef.current.push({
                role: "user",
                content: `I ran the shell command "$ ${cmd}" and got this output:\n\n${truncOutput}\n\nEach line above is a file or folder name. List them and briefly describe what you see. Do not say this is a screenshot — these are actual file/folder names returned by the command.`,
              });
              setIsStreaming(true);
              setStreamingText("");
              onStatusChange("streaming");
              abortRef.current = new AbortController();
              let fullText = "";
              await client.streamChat(messagesRef.current, {
                onToken: (token) => { setSpinnerMsg(""); fullText += token; setStreamingText(fullText); },
                onDone: (stats) => {
                  setSpinnerMsg("");
                  messagesRef.current.push({ role: "assistant", content: fullText });
                  setEntries((prev) => [...prev, { role: "assistant", content: fullText }]);
                  setIsStreaming(false);
                  setStreamingText("");
                  onStatsUpdate(stats);
                  onStatusChange("connected");
                },
                onError: (err) => {
                  setSpinnerMsg("");
                  setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${err}` }]);
                  setIsStreaming(false);
                  setStreamingText("");
                  onStatusChange("connected");
                },
              }, abortRef.current.signal);
            } catch (e: any) {
              setSpinnerMsg("");
              setEntries((prev) => [
                ...prev,
                {
                  role: "error",
                  content: `◆ Error: ${e.message?.split("\n")[0]}`,
                },
              ]);
            }
            return;
          }

          case "/search": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /search <query>" },
              ]);
              return;
            }
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `/search ${args}` },
            ]);
            setSpinnerMsg("Searching the web...");
            try {
              const safeQuery = args.replace(/[`$\\'"]/g, "");
              const isNews = /news|today|latest|recent|happened|breaking|update/i.test(args);
              const searchResults = execSync(
                `python3 -c "
import sys
from ddgs import DDGS
from datetime import datetime
query = sys.argv[1]
is_news = sys.argv[2] == '1'
results = []
with DDGS() as d:
    if is_news:
        for r in d.news(query, max_results=8):
            date = r.get('date', '')[:10]
            src = r.get('source', '')
            results.append('- [' + date + '] ' + r.get('title','') + ' (' + src + '): ' + r.get('body',''))
    else:
        dated_query = query + ' ' + datetime.now().strftime('%B %Y')
        for r in d.text(dated_query, max_results=8):
            results.append('- ' + r.get('title','') + ': ' + r.get('body',''))
print(chr(10).join(results))
" "${safeQuery}" "${isNews ? '1' : '0'}"`,
                { encoding: "utf-8", timeout: 15000 }
              ).trim();
              setSpinnerMsg("");
              if (searchResults) {
                setEntries((prev) => [
                  ...prev,
                  { role: "info", content: `◆ Found ${searchResults.split('\n').length} results:\n${searchResults.split('\n').map(l => '  ' + l.slice(0, 120)).join('\n')}` },
                ]);
                setSpinnerMsg("Synthesizing answer...");
                // Send results to LLM for synthesis
                const today = new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
                messagesRef.current.push({
                  role: "user",
                  content: `Today is ${today}. Summarize the key points from these search results. Be specific — include names, dates, and facts. Do not just list the sources.\n\nSearch results:\n${searchResults}\n\nUser's question: ${args}`,
                });
                // Stream LLM response (falls through to streaming below)
                setIsStreaming(true);
                setStreamingText("");
                onStatusChange("streaming");
                abortRef.current = new AbortController();
                let fullText = "";
                await client.streamChat(messagesRef.current, {
                  onToken: (token) => { setSpinnerMsg(""); fullText += token; setStreamingText(fullText); },
                  onDone: (stats) => {
                    setSpinnerMsg("");
                    messagesRef.current.push({ role: "assistant", content: fullText });
                    setEntries((prev) => [...prev, { role: "assistant", content: fullText }]);
                    setIsStreaming(false);
                    setStreamingText("");
                    onStatsUpdate(stats);
                    onStatusChange("connected");
                  },
                  onError: (err) => {
                    setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${err}` }]);
                    setIsStreaming(false);
                    setStreamingText("");
                    onStatusChange("connected");
                  },
                }, abortRef.current.signal);
                return;
              } else {
                setEntries((prev) => [...prev, { role: "info", content: "◆ No results found." }]);
              }
            } catch {
              setEntries((prev) => [
                ...prev,
                { role: "info", content: "◆ Search failed. Check: python3 search.py test" },
              ]);
            }
            return;
          }

          case "/image": {
            if (!args) {
              setEntries((prev) => [
                ...prev,
                { role: "error", content: "◆ Usage: /image <path>" },
              ]);
              return;
            }
            // Send as a chat message asking to describe
            setEntries((prev) => [
              ...prev,
              { role: "user", content: `[Describe image: ${args}]` },
            ]);
            messagesRef.current.push({
              role: "user",
              content: `Please describe what you think might be in an image at path: ${args}. Note: I cannot actually see images, but I can help discuss image-related topics.`,
            });
            break; // fall through to streaming
          }

          case "/screenshot": {
            setEntries((prev) => [
              ...prev,
              { role: "user", content: "[Taking screenshot...]" },
            ]);
            try {
              execSync("screencapture -x /tmp/tinybit-screenshot.png", {
                timeout: 5000,
              });
              setEntries((prev) => [
                ...prev,
                {
                  role: "info",
                  content: "◆ Screenshot saved to /tmp/tinybit-screenshot.png",
                },
              ]);
            } catch {
              setEntries((prev) => [
                ...prev,
                {
                  role: "error",
                  content: "◆ Screenshot failed (requires macOS)",
                },
              ]);
            }
            return;
          }

          default:
            setEntries((prev) => [
              ...prev,
              {
                role: "error",
                content: `◆ Unknown command: ${cmd}. Type /help for commands.`,
              },
            ]);
            return;
        }
      }

      // Regular chat message
      if (!trimmed.startsWith("/")) {
        setEntries((prev) => [...prev, { role: "user", content: trimmed }]);
        messagesRef.current.push({ role: "user", content: trimmed });
      }

      // Agent loop: call LLM, check for tool calls, execute, feed back, repeat
      let loopCount = 0;
      const MAX_LOOPS = 5;

      while (loopCount < MAX_LOOPS) {
        loopCount++;
        setSpinnerMsg(loopCount === 1 ? "Thinking..." : "Processing tool results...");

        // Get non-streaming response to check for tool calls
        let response = "";
        try {
          response = await client.chat(messagesRef.current, 500);
        } catch (e: any) {
          setSpinnerMsg("");
          setEntries((prev) => [...prev, { role: "error", content: `◆ Error: ${e.message}` }]);
          break;
        }

        setSpinnerMsg("");

        // Check for tool_call in response
        const toolCallMatch = response.match(/<tool_call>([\s\S]*?)<\/tool_call>/);

        if (!toolCallMatch) {
          // No tool call — this is the final text response. Stream it for nice display.
          messagesRef.current.push({ role: "assistant", content: response });
          setEntries((prev) => [...prev, { role: "assistant", content: response }]);
          break;
        }

        // Parse and execute tool call
        try {
          const toolCall = JSON.parse(toolCallMatch[1]);
          const toolName = toolCall.name;
          const toolArgs = toolCall.arguments || {};

          // Show what the model is doing
          messagesRef.current.push({ role: "assistant", content: response });

          if (toolName === "shell") {
            const cmd = toolArgs.command;
            setSpinnerMsg(`Running: $ ${cmd}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ Running: $ ${cmd}` }]);

            let output = "";
            try {
              output = execSync(cmd, { encoding: "utf-8", timeout: 10000, maxBuffer: 1024 * 1024 }).trim();
            } catch (e: any) {
              output = `Error: ${e.message?.split("\n")[0]}`;
            }
            const truncOutput = output.split("\n").slice(0, 30).join("\n");
            setSpinnerMsg("");
            setEntries((prev) => [...prev, { role: "info", content: `◆ Output:\n${truncOutput}` }]);

            // Feed result back to model
            messagesRef.current.push({
              role: "user",
              content: `<tool_response>{"name": "shell", "output": ${JSON.stringify(truncOutput)}}</tool_response>`,
            });

          } else if (toolName === "web_search") {
            const query = toolArgs.query;
            setSpinnerMsg(`Searching: ${query}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ Searching: ${query}` }]);

            let results = "";
            try {
              const safeQuery = query.replace(/[`$\\'"]/g, "");
              const isNews = /news|today|latest|happened/i.test(query);
              results = execSync(
                `python3 -c "
import sys
from ddgs import DDGS
from datetime import datetime
q = sys.argv[1]
is_news = sys.argv[2] == '1'
res = []
with DDGS() as d:
    if is_news:
        for r in d.news(q, max_results=5):
            res.append(r.get('date','')[:10] + ' - ' + r.get('title','') + ': ' + r.get('body',''))
    else:
        for r in d.text(q + ' ' + datetime.now().strftime('%B %Y'), max_results=5):
            res.append(r.get('title','') + ': ' + r.get('body',''))
print(chr(10).join(res))
" "${safeQuery}" "${isNews ? '1' : '0'}"`,
                { encoding: "utf-8", timeout: 15000 }
              ).trim();
            } catch {
              results = "Search failed";
            }
            setSpinnerMsg("");
            setEntries((prev) => [...prev, { role: "info", content: `◆ Results:\n${results.split('\n').map(l => '  ' + l.slice(0, 100)).join('\n')}` }]);

            messagesRef.current.push({
              role: "user",
              content: `<tool_response>{"name": "web_search", "output": ${JSON.stringify(results)}}</tool_response>`,
            });

          } else if (toolName === "read_file") {
            const filePath = toolArgs.path.replace(/^~/, process.env.HOME || "");
            const isImage = /\.(png|jpg|jpeg|gif|webp|bmp|svg|heic)$/i.test(filePath);
            const isPDF = /\.pdf$/i.test(filePath);
            const isDoc = /\.(doc|docx|rtf|html|htm)$/i.test(filePath);

            setSpinnerMsg(isImage ? `Analyzing image: ${filePath}` : isPDF ? `Parsing PDF: ${filePath}` : `Reading: ${filePath}`);
            setEntries((prev) => [...prev, { role: "info", content: `◆ ${isImage ? "Analyzing" : "Reading"}: ${filePath}` }]);

            let content = "";
            try {
              if (isImage) {
                // Use vision API — encode image as base64 and send to the model
                const b64 = execSync(`base64 -i "${filePath}"`, { encoding: "utf-8", timeout: 10000 }).trim().replace(/\n/g, "");
                const ext = filePath.split(".").pop()?.toLowerCase() || "png";
                const mime = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif", webp: "image/webp", heic: "image/heic" }[ext] || "image/png";

                // Send directly as a vision message — break out of tool loop for this
                setSpinnerMsg("Model analyzing image...");
                const visionResp = await client.chat([
                  { role: "user", content: [
                    { type: "image_url", image_url: { url: `data:${mime};base64,${b64}` } },
                    { type: "text", text: "Describe what you see in this image in detail." },
                  ] as any },
                ], 300);

                setSpinnerMsg("");
                messagesRef.current.push({ role: "assistant", content: visionResp });
                setEntries((prev) => [...prev, { role: "assistant", content: visionResp }]);
                break; // Exit tool loop — vision response is final

              } else if (isPDF) {
                // macOS textutil can't do PDF, try python
                try {
                  content = execSync(`python3 -c "
import subprocess
result = subprocess.run(['mdimport', '-d2', '${filePath}'], capture_output=True, text=True, timeout=10)
print(result.stderr[:3000])
"`, { encoding: "utf-8", timeout: 15000 }).trim();
                  if (!content || content.length < 50) {
                    // Fallback: try strings
                    content = execSync(`strings "${filePath}" | head -100`, { encoding: "utf-8", timeout: 5000 }).trim();
                  }
                } catch {
                  content = "Could not parse PDF. Try: pip install pymupdf";
                }

              } else if (isDoc) {
                // macOS textutil converts doc/docx/rtf/html to text
                content = execSync(`textutil -convert txt -stdout "${filePath}" | head -100`, { encoding: "utf-8", timeout: 10000 }).trim();

              } else {
                // Regular text file
                content = execSync(`head -80 "${filePath}"`, { encoding: "utf-8", timeout: 5000 }).trim();
              }
            } catch (err: any) {
              content = `Error reading file: ${err.message?.split("\n")[0]}`;
            }
            setSpinnerMsg("");

            messagesRef.current.push({
              role: "user",
              content: `<tool_response>{"name": "read_file", "output": ${JSON.stringify(content)}}</tool_response>`,
            });

          } else {
            // Unknown tool
            setEntries((prev) => [...prev, { role: "error", content: `◆ Unknown tool: ${toolName}` }]);
            break;
          }
        } catch (e: any) {
          setEntries((prev) => [...prev, { role: "error", content: `◆ Tool parse error: ${e.message}` }]);
          // Still show the raw response
          messagesRef.current.push({ role: "assistant", content: response });
          setEntries((prev) => [...prev, { role: "assistant", content: response.replace(/<tool_call>[\s\S]*?<\/tool_call>/, '').trim() }]);
          break;
        }
      }

      onStatusChange("connected");
      setIsStreaming(false);
      setStreamingText("");
      setSpinnerMsg("");
    },
    [client, onQuit, onStatsUpdate, onStatusChange]
  );

  const getEntryColor = (role: ChatEntry["role"]) => {
    switch (role) {
      case "user":
        return "#FFBF00";
      case "assistant":
        return "#33FF33";
      case "system":
        return "#33FF33";
      case "info":
        return "#33FF33";
      case "error":
        return "red";
    }
  };

  const getEntryPrefix = (role: ChatEntry["role"]) => {
    switch (role) {
      case "user":
        return "▶ you";
      case "assistant":
        return " tiny bit";
      case "system":
        return "";
      case "info":
        return "";
      case "error":
        return "⚠";
    }
  };

  return (
    <Box flexDirection="column" flexGrow={1}>
      {/* Chat history */}
      <Box flexDirection="column" flexGrow={1} paddingX={1}>
        {visibleEntries.map((entry, i) => (
          <Box key={i} flexDirection="column" marginBottom={0}>
            {entry.role !== "system" && entry.role !== "info" && entry.role !== "error" && (
              <Text color={getEntryColor(entry.role)} bold dimColor={entry.role === "assistant"}>
                {getEntryPrefix(entry.role)}:
              </Text>
            )}
            <Text
              color={getEntryColor(entry.role)}
              wrap="wrap"
            >
              {entry.role === "user" ? "  " : ""}{entry.content}
            </Text>
          </Box>
        ))}

        {/* Streaming response */}
        {spinnerMsg && !isStreaming && (
          <Box paddingLeft={2}>
            <RetroSpinner type="dots" label={spinnerMsg} />
          </Box>
        )}
        {isStreaming && (
          <Box flexDirection="column">
            <Text color="#33FF33" bold dimColor>
               tiny bit:
            </Text>
            <Text color="#33FF33" wrap="wrap">
              {streamingText}
              <Text color="#33FF33">█</Text>
            </Text>
            <Box marginTop={0}>
              <RetroSpinner
                label="generating..."
                type="dots"
              />
            </Box>
          </Box>
        )}
      </Box>

      {/* Input area */}
      <Box paddingX={1}>
        <Text color="#33FF33" dimColor>
          ╠══════════════════════════════════════════════════════════════════════════════╣
        </Text>
      </Box>
      <Box paddingX={1}>
        <Text color="#FFBF00" bold>
          ▶{" "}
        </Text>
        {isStreaming ? (
          <Text color="#33FF33" dimColor>
            (streaming... press ESC to stop)
          </Text>
        ) : (
          <TextInput
            value={input}
            onChange={(val) => {
              setInput(val);
              setShowCommands(val === "/");
            }}
            onSubmit={(val) => {
              setShowCommands(false);
              handleSubmit(val);
            }}
            placeholder="Type a message or /help..."
          />
        )}
      </Box>
      {showCommands && (
        <Box flexDirection="column" paddingLeft={4}>
          <Text color="#1A8C1A">{"─".repeat(40)}</Text>
          <Text color="#33FF33"> /search {"<query>"}   <Text color="#1A8C1A">web search + AI synthesis</Text></Text>
          <Text color="#33FF33"> /image {"<path>"}     <Text color="#1A8C1A">describe an image</Text></Text>
          <Text color="#33FF33"> /screenshot       <Text color="#1A8C1A">capture + analyze screen</Text></Text>
          <Text color="#33FF33"> /shell {"<cmd>"}      <Text color="#1A8C1A">run a shell command</Text></Text>
          <Text color="#33FF33"> /compare {"<prompt>"}  <Text color="#1A8C1A">compare Bonsai vs 9B</Text></Text>
          <Text color="#33FF33"> /stats            <Text color="#1A8C1A">show performance stats</Text></Text>
          <Text color="#33FF33"> /clear            <Text color="#1A8C1A">clear conversation</Text></Text>
          <Text color="#33FF33"> /help             <Text color="#1A8C1A">show all commands</Text></Text>
          <Text color="#33FF33"> /quit             <Text color="#1A8C1A">exit</Text></Text>
          <Text color="#1A8C1A">{"─".repeat(40)}</Text>
        </Box>
      )}
    </Box>
  );
};
