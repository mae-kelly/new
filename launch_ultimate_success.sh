#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${RED}🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥${NC}"
echo -e "${RED}🔥                                                            🔥${NC}"
echo -e "${RED}🔥           ULTIMATE 20,000X SUCCESS SYSTEM                  🔥${NC}"
echo -e "${RED}🔥                                                            🔥${NC}"
echo -e "${RED}🔥              FINAL EVOLUTION ACTIVATED                     🔥${NC}"
echo -e "${RED}🔥                                                            🔥${NC}"
echo -e "${RED}🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥${NC}"
echo ""

echo -e "${CYAN}Choose your path to ultimate success:${NC}"
echo ""
echo -e "${GREEN}1. 🚀 Original Mega Success System${NC}"
echo "   - Proven strategy with excellent results"
echo "   - Safe and reliable"
echo ""
echo -e "${YELLOW}2. 🔥 Ultra Aggressive System${NC}"
echo "   - Maximum risk/reward ratio"
echo "   - Advanced MEV strategies"
echo ""
echo -e "${RED}3. 💀 ULTIMATE SUCCESS COORDINATOR${NC}"
echo "   - ALL SYSTEMS COMBINED"
echo "   - MAXIMUM POWER MODE"
echo "   - FINAL EVOLUTION"
echo ""
echo -e "${PURPLE}4. 🧠 AI-Optimized Custom System${NC}"
echo "   - AI continuously optimizes parameters"
echo "   - Adapts to market conditions"
echo ""

read -p "Enter your choice (1-4): " CHOICE

echo ""
read -p "💰 Enter starting balance (default: 10): " BALANCE
BALANCE=${BALANCE:-10}

# Create all necessary directories
mkdir -p {logs,ai_systems,exploit_systems,profit_maximizers}

echo ""
echo -e "${BLUE}🚀 Preparing for launch...${NC}"
sleep 2

case $CHOICE in
    1)
        echo -e "${GREEN}🚀 Launching Mega Success System...${NC}"
        echo -e "${GREEN}✅ Reliable and proven strategy${NC}"
        python3 mega_strategies/mega_orchestrator.py $BALANCE
        ;;
    2)
        echo -e "${YELLOW}🔥 Launching Ultra Aggressive System...${NC}"
        echo -e "${YELLOW}⚠️  High risk/high reward mode${NC}"
        python3 profit_maximizers/ultimate_coordinator.py $BALANCE
        ;;
    3)
        echo -e "${RED}💀 ULTIMATE SUCCESS COORDINATOR LAUNCHING...${NC}"
        echo -e "${RED}🔥 ALL SYSTEMS: MAXIMUM POWER${NC}"
        echo -e "${RED}⚡ FINAL EVOLUTION ENGAGED${NC}"
        echo ""
        echo -e "${CYAN}This is the most powerful configuration.${NC}"
        echo -e "${CYAN}All systems will run simultaneously.${NC}"
        echo ""
        read -p "Ready to unleash ultimate power? (y/N): " CONFIRM
        if [[ $CONFIRM =~ ^[Yy]$ ]]; then
            python3 ultimate_success_coordinator.py $BALANCE
        else
            echo -e "${GREEN}Running safe Mega Success System instead.${NC}"
            python3 mega_strategies/mega_orchestrator.py $BALANCE
        fi
        ;;
    4)
        echo -e "${PURPLE}🧠 Launching AI-Optimized System...${NC}"
        echo -e "${PURPLE}🤖 AI will continuously optimize performance${NC}"
        # Run AI optimizer alongside mega orchestrator
        python3 mega_strategies/mega_orchestrator.py $BALANCE &
        python3 ai_systems/ai_auto_optimizer.py &
        wait
        ;;
    *)
        echo -e "${RED}Invalid choice. Launching default Mega Success System.${NC}"
        python3 mega_strategies/mega_orchestrator.py $BALANCE
        ;;
esac
