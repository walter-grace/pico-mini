import React, { useState, useEffect } from "react";
import { Text } from "ink";
import { CRT } from "../theme.js";

// CRT-style spinner animations
const SPINNERS = {
  classic: ["|", "/", "-", "\\"],
  dots: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
  blocks: ["░", "▒", "▓", "█", "▓", "▒"],
  pulse: ["·", "•", "●", "•"],
  bar: ["[    ]", "[=   ]", "[==  ]", "[=== ]", "[====]", "[ ===]", "[  ==]", "[   =]"],
};

type SpinnerType = keyof typeof SPINNERS;

interface RetroSpinnerProps {
  type?: SpinnerType;
  label?: string;
}

export const RetroSpinner: React.FC<RetroSpinnerProps> = ({
  type = "dots",
  label = "",
}) => {
  const [frame, setFrame] = useState(0);
  const frames = SPINNERS[type];

  useEffect(() => {
    const t = setInterval(() => setFrame(f => (f + 1) % frames.length), 100);
    return () => clearInterval(t);
  }, [frames.length]);

  return (
    <Text color={CRT.green}>
      {frames[frame]}{label ? ` ${label}` : ""}
    </Text>
  );
};
