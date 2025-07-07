#!/bin/bash
# Simple launcher for Ultimate Trading System

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 ULTIMATE TRADING SYSTEM${NC}"
echo -e "${BLUE}Starting system...${NC}"

# Check if setup was completed
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ System not set up. Run setup first.${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Show current status
echo -e "${GREEN}📊 Current Status:${NC}"
python scripts/utils/monitor.py

echo ""
echo -e "${GREEN}✅ System ready!${NC}"
echo -e "${GREEN}🎯 Next: Implement your trading strategies${NC}"
echo -e "${GREEN}📚 Check the scripts/ directory for examples${NC}"
