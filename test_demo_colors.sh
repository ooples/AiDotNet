#!/bin/bash

# Test script to verify color output without interaction

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR="/home/ooples/AiDotNet"
AGENTS_DIR="$PROJECT_DIR/BuildFixAgents"

echo -e "${BLUE}=== Testing Color Output ===${NC}"
echo
echo -e "${GREEN}✓ Green text works${NC}"
echo -e "${RED}✗ Red text works${NC}"
echo -e "${YELLOW}⚠ Yellow text works${NC}"
echo -e "${CYAN}ℹ Cyan text works${NC}"
echo
echo -e "\n${GREEN}Ready to run the full process?${NC}"
echo
echo -e "Execute: ${YELLOW}./test_auto_agent_process.sh${NC}"
echo
echo "Or run individual components:"
echo -e "  1. ${YELLOW}cd $AGENTS_DIR${NC}"
echo -e "  2. ${YELLOW}./build_analyzer.sh ../build_output.txt${NC}"
echo -e "  3. ${YELLOW}./autofix.sh all${NC}"
echo
echo -e "${BLUE}=== Color test complete ===${NC}"