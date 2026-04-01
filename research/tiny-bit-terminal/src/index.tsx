#!/usr/bin/env node
import React from "react";
import { render } from "ink";
import { App } from "./App.js";

// Parse CLI args
function parseArgs(): { serverUrl: string } {
  const args = process.argv.slice(2);
  let serverUrl = "http://localhost:8080";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--server" && args[i + 1]) {
      serverUrl = args[i + 1];
      i++;
    } else if (args[i]?.startsWith("--server=")) {
      serverUrl = args[i].split("=")[1];
    } else if (args[i] === "--help" || args[i] === "-h") {
      console.log(`
╔══════════════════════════════════════════════════╗
║   tiny bit ── Local AI Terminal Agent           ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Usage:                                          ║
║    npx tsx src/index.tsx [options]                ║
║                                                  ║
║  Options:                                        ║
║    --server <url>  llama-server URL              ║
║                    (default: localhost:8080)      ║
║    --help, -h      Show this help                ║
║                                                  ║
║  Commands (in chat):                             ║
║    /search <q>     Search files                  ║
║    /image <path>   Describe image                ║
║    /screenshot     Capture screen                ║
║    /shell <cmd>    Run shell command             ║
║    /stats          Server statistics             ║
║    /clear          Clear chat                    ║
║    /quit           Exit                          ║
║                                                  ║
╚══════════════════════════════════════════════════╝
`);
      process.exit(0);
    }
  }

  return { serverUrl };
}

const { serverUrl } = parseArgs();

// Render with ink (fullscreen mode)
const { waitUntilExit } = render(React.createElement(App, { serverUrl }), {
  exitOnCtrlC: true,
});

waitUntilExit().then(() => {
  process.exit(0);
});
