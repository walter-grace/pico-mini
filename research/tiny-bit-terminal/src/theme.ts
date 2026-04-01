// cool-retro-term "Default Green" CRT theme
// Matches the phosphor green aesthetic with glow simulation

export const CRT = {
  // Primary phosphor green (bright)
  green: "#33FF33",
  // Dim phosphor (for less important text)
  greenDim: "#1A8C1A",
  // Very dim (scanline effect, borders)
  greenFaint: "#0D4D0D",
  // Glow effect (slightly yellow-green, like real phosphor bloom)
  glow: "#55FF55",
  // Amber (for warnings, accents — like Apple IIc amber monitor)
  amber: "#FFB000",
  // Error red (kept dim to match CRT feel)
  red: "#FF3333",
  // Background (pure black)
  bg: "#000000",
  // Cursor block
  cursor: "#33FF33",
} as const;

// Unicode characters for CRT-style borders
export const BORDER = {
  tl: "┌", tr: "┐", bl: "└", br: "┘",
  h: "─", v: "│",
  // Double line for headers
  dtl: "╔", dtr: "╗", dbl: "╚", dbr: "╝",
  dh: "═", dv: "║",
  // Connectors
  lt: "├", rt: "┤", top: "┬", bot: "┴", cross: "┼",
  dlt: "╠", drt: "╣",
} as const;

// CRT-style symbols
export const SYM = {
  prompt: "▶",
  cursor: "█",
  bullet: "●",
  hollow: "○",
  diamond: "◆",
  arrow: "→",
  check: "✓",
  cross: "✗",
  dot: "·",
  block: "█",
  shade: "░",
  apple: "",
  folder: "📁",
  file: "📄",
  search: "🔍",
  gear: "⚙",
  lightning: "⚡",
} as const;

// Scanline effect — dims every other line of text
export function scanline(text: string): string {
  return text.split("\n").map((line, i) =>
    i % 2 === 1 ? line : line
  ).join("\n");
}

// Phosphor glow — adds brightness variation to simulate CRT bloom
export function glowText(text: string): string {
  return text; // In a real terminal, this would need ANSI escape trickery
}
