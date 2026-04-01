import React from "react";
import { Box, Text } from "ink";
import { CRT, SYM, BORDER } from "../theme.js";

interface StatusBarProps {
  modelName: string;
  tokPerSec: number;
  totalTokens: number;
  serverUrl: string;
  status: "connected" | "disconnected" | "streaming";
}

export const StatusBar: React.FC<StatusBarProps> = ({
  modelName,
  tokPerSec,
  totalTokens,
  serverUrl,
  status,
}) => {
  const statusIcon = status === "connected" ? SYM.bullet
    : status === "streaming" ? SYM.lightning
    : SYM.hollow;
  const statusColor = status === "connected" ? CRT.green
    : status === "streaming" ? CRT.glow
    : CRT.red;

  const memMB = Math.round(process.memoryUsage().rss / 1024 / 1024);

  return (
    <Box flexDirection="column">
      <Text color={CRT.greenDim}>{BORDER.dlt}{BORDER.dh.repeat(78)}{BORDER.drt}</Text>
      <Box>
        <Text color={CRT.greenDim}>{BORDER.dv} </Text>
        <Text color={statusColor}>{statusIcon}</Text>
        <Text color={CRT.green}> {modelName}</Text>
        <Text color={CRT.greenDim}> {BORDER.v} </Text>
        <Text color={CRT.green}>{tokPerSec > 0 ? `${tokPerSec} tok/s` : "idle"}</Text>
        <Text color={CRT.greenDim}> {BORDER.v} </Text>
        <Text color={CRT.greenDim}>{totalTokens} tokens</Text>
        <Text color={CRT.greenDim}> {BORDER.v} </Text>
        <Text color={CRT.greenDim}>{memMB}MB</Text>
        <Text color={CRT.greenDim}> {BORDER.v} </Text>
        <Text color={CRT.greenDim}>{serverUrl}</Text>
        <Text color={CRT.greenDim}> {BORDER.dv}</Text>
      </Box>
    </Box>
  );
};
