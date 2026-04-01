import React from "react";
import { Box, Text } from "ink";
import { CRT, BORDER } from "../theme.js";

interface RetroBoxProps {
  title?: string;
  children: React.ReactNode;
  width?: number;
  dim?: boolean;
}

export const RetroBox: React.FC<RetroBoxProps> = ({
  title,
  children,
  width = 80,
  dim = false,
}) => {
  const bc = dim ? CRT.greenFaint : CRT.greenDim;
  const tc = CRT.green;
  const innerWidth = width - 4;

  return (
    <Box flexDirection="column">
      <Box>
        <Text color={bc}>{BORDER.dtl}</Text>
        {title ? (
          <>
            <Text color={bc}>{BORDER.dh}{BORDER.dh} </Text>
            <Text color={tc} bold>{title}</Text>
            <Text color={bc}> {BORDER.dh.repeat(Math.max(0, innerWidth - title.length - 2))}</Text>
          </>
        ) : (
          <Text color={bc}>{BORDER.dh.repeat(innerWidth + 2)}</Text>
        )}
        <Text color={bc}>{BORDER.dtr}</Text>
      </Box>
      <Box>
        <Text color={bc}>{BORDER.dv} </Text>
        <Box flexGrow={1}>
          {children}
        </Box>
        <Text color={bc}> {BORDER.dv}</Text>
      </Box>
      <Box>
        <Text color={bc}>{BORDER.dbl}{BORDER.dh.repeat(innerWidth + 2)}{BORDER.dbr}</Text>
      </Box>
    </Box>
  );
};
