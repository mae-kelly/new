#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}📊 20,000X PERFORMANCE MONITOR${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)

if [ -f "$LATEST_LOG" ]; then
    echo -e "${GREEN}🚀 LIVE PERFORMANCE DATA:${NC}"
    echo ""
    
    echo -e "${BLUE}📈 Latest Status:${NC}"
    tail -20 "$LATEST_LOG" | grep -E "(Balance|Return|PROFIT|WIN)" | tail -5
    echo ""
    
    echo -e "${YELLOW}💰 Recent Profitable Trades:${NC}"
    tail -50 "$LATEST_LOG" | grep -E "(PROFIT|WIN)" | tail -10
    
else
    echo -e "${YELLOW}⚠️  No active trading sessions found.${NC}"
    echo -e "${GREEN}💡 Start trading with: ./deploy_20000x_success.sh${NC}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
