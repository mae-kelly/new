GREEN='\033[0
YELLOW='\033[1
RED='\033[0
NC='\033[0m'
echo -e "${GREEN}🚀 Starting Crypto Trading Bot${NC}"
if [ ! -f .env ]
    echo -e "${RED}❌ .env file not found. Run ./setup_real_bot.sh first${NC}"
    exit 1
fi
if ! grep -q "your_base58_private_key_here" .env
    echo -e "${GREEN}✅ Wallet configured${NC}"
else
    echo -e "${YELLOW}⚠️  Wallet not configured. Creating new wallet...${NC}"
    python3 setup_wallet.py
fi
echo
echo -e "${YELLOW}Select trading mode:${NC}"
echo "1. Paper Trading (Safe - No real money)"
echo "2. Live Trading (Real money - High risk)"
echo
read -p "Enter choice (1 or 2): " choice
if [ "$choice" = "2" ]
    echo -e "${RED}⚠️  WARNING: LIVE TRADING MODE${NC}"
    echo -e "${RED}💸 You can lose real money!${NC}"
    echo
    read -p "Type 'YES' to continue with live trading: " confirm
    if [ "$confirm" != "YES" ]
        echo -e "${GREEN}👍 Using paper trading mode${NC}"
        choice="1"
    fi
fi
if [ "$choice" = "2" ]
    sed -i 's/PAPER_TRADING=true/PAPER_TRADING=false/' .env
    echo -e "${RED}🔴 LIVE TRADING MODE ACTIVATED${NC}"
else
    sed -i 's/PAPER_TRADING=false/PAPER_TRADING=true/' .env
    echo -e "${GREEN}🟢 PAPER TRADING MODE ACTIVATED${NC}"
fi
echo
echo -e "${GREEN}🚀 Starting bot...${NC}"
python3 real_trading_bot.py
