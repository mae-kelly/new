#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${RED}🔥🔥🔥 20,000X SUCCESS DEPLOYMENT 🔥🔥🔥${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Choose your trading intensity:${NC}"
echo ""
echo -e "${GREEN}1. 🚀 Mega Success System (Recommended)${NC}"
echo -e "${RED}2. 🔥 Ultra Aggressive System (Maximum Risk/Reward)${NC}"
echo ""

read -p "Enter choice (1 or 2): " CHOICE

read -p "💰 Enter starting balance (default: 10): " BALANCE
BALANCE=${BALANCE:-10}

mkdir -p logs

case $CHOICE in
    1)
        echo -e "${GREEN}🚀 Launching Mega Success System...${NC}"
        python3 mega_strategies/mega_orchestrator.py $BALANCE
        ;;
    2)
        echo -e "${RED}🔥 Launching Ultra Aggressive System...${NC}"
        python3 profit_maximizers/ultimate_coordinator.py $BALANCE
        ;;
    *)
        echo -e "${GREEN}Running default Mega Success System.${NC}"
        python3 mega_strategies/mega_orchestrator.py $BALANCE
        ;;
esac
