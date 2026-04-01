import React, { useState, useEffect } from "react";
import { Box, Text } from "ink";
import { CRT, SYM } from "../theme.js";

interface BootScreenProps {
  onComplete: () => void;
}

const MACINTOSH = [
  "          ______________",
  "         /             /|",
  "        /             / |",
  "       /____________ /  |",
  "      | ___________ |   |",
  "      ||           ||   |",
  "      ||  tiny bit ||   |",
  "      ||           ||   |",
  "      ||___________||   |",
  "      |   _______   |  /",
  "     /|  (_______)  | /",
  "    ( |_____________|/",
  "     \\",
  " .=======================.",
  " | ::::::::::::::::  ::: |",
  " | ::::::::::::::[]  ::: |",
  " |   -----------     ::: |",
  " `-----------------------'",
];

const BOOT_LINES = [
  "tiny bit v1.0 — local AI terminal",
  "",
  "Testing memory............ OK",
  "Loading inference engine... OK",
  "Initializing search....... OK",
  "Connecting to server...... OK",
  "",
  "Welcome to tiny bit.",
];

export const BootScreen: React.FC<BootScreenProps> = ({ onComplete }) => {
  const [phase, setPhase] = useState(0);
  const [bootLine, setBootLine] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 2000);
    return () => clearTimeout(t1);
  }, []);

  useEffect(() => {
    if (phase !== 1) return;
    if (bootLine >= BOOT_LINES.length) {
      const t = setTimeout(() => onComplete(), 1000);
      return () => clearTimeout(t);
    }
    const delay = BOOT_LINES[bootLine] === "" ? 100 : 300;
    const t = setTimeout(() => setBootLine(b => b + 1), delay);
    return () => clearTimeout(t);
  }, [phase, bootLine, onComplete]);

  if (phase === 0) {
    return (
      <Box flexDirection="column" alignItems="center" justifyContent="center" height={24}>
        <Text> </Text>
        {MACINTOSH.map((line, i) => (
          <Text key={i} color={i % 2 === 0 ? CRT.green : CRT.greenDim}>{line}</Text>
        ))}
        <Text> </Text>
        <Text color={CRT.greenDim}>starting up...</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingLeft={2} paddingTop={1}>
      <Text color={CRT.green} bold>
        {SYM.diamond} tiny bit {SYM.diamond}
      </Text>
      <Text color={CRT.greenDim}>{"─".repeat(40)}</Text>
      {BOOT_LINES.slice(0, bootLine).map((line, i) => (
        <Text key={i} color={
          line.includes("Welcome") ? CRT.glow :
          line.includes("OK") ? CRT.green :
          CRT.greenDim
        }>
          {line ? `  ${line}` : ""}
        </Text>
      ))}
      {bootLine < BOOT_LINES.length && (
        <Text color={CRT.green}>  {SYM.cursor}</Text>
      )}
    </Box>
  );
};
