#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🚀 LAUNCHING MEGA SUCCESS SYSTEM${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"

read -p "💰 Enter starting balance (default: 10): " BALANCE
BALANCE=${BALANCE:-10}

echo -e "${GREEN}🚀 Starting with $${BALANCE}...${NC}"
mkdir -p logs

python3 mega_strategies/mega_orchestrator.py $BALANCE
