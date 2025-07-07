#!/bin/bash

# start_bot.sh - Start the trading bot

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Crypto Trading Bot${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found. Run ./setup_real_bot.sh first${NC}"
    exit 1
fi

# Check if wallet is configured
if ! grep -q "your_base58_private_key_here" .env; then
    echo -e "${GREEN}✅ Wallet configured${NC}"
else
    echo -e "${YELLOW}⚠️  Wallet not configured. Creating new wallet...${NC}"
    python3 setup_wallet.py
fi

# Ask about trading mode
echo
echo -e "${YELLOW}Select trading mode:${NC}"
echo "1. Paper Trading (Safe - No real money)"
echo "2. Live Trading (Real money - High risk)"
echo
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "2" ]; then
    echo -e "${RED}⚠️  WARNING: LIVE TRADING MODE${NC}"
    echo -e "${RED}💸 You can lose real money!${NC}"
    echo
    read -p "Type 'YES' to continue with live trading: " confirm
    if [ "$confirm" != "YES" ]; then
        echo -e "${GREEN}👍 Using paper trading mode${NC}"
        choice="1"
    fi
fi

# Update .env file with trading mode
if [ "$choice" = "2" ]; then
    sed -i 's/PAPER_TRADING=true/PAPER_TRADING=false/' .env
    echo -e "${RED}🔴 LIVE TRADING MODE ACTIVATED${NC}"
else
    sed -i 's/PAPER_TRADING=false/PAPER_TRADING=true/' .env
    echo -e "${GREEN}🟢 PAPER TRADING MODE ACTIVATED${NC}"
fi

echo
echo -e "${GREEN}🚀 Starting bot...${NC}"
python3 real_trading_bot.py
