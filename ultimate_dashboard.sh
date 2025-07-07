#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Function to display header
show_header() {
    clear
    echo -e "${RED}🔥🔥🔥 ULTIMATE SUCCESS DASHBOARD 🔥🔥🔥${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Last Updated: $(date)${NC}"
    echo ""
}

# Function to show live performance
show_live_performance() {
    echo -e "${GREEN}📊 LIVE PERFORMANCE DATA:${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
    
    # Find all log files
    LOGS=$(ls -t logs/*.log 2>/dev/null)
    
    if [ -n "$LOGS" ]; then
        # Show latest balance from all systems
        for log in $LOGS; do
            LATEST_BALANCE=$(tail -20 "$log" | grep -E "Balance.*:" | tail -1)
            if [ -n "$LATEST_BALANCE" ]; then
                SYSTEM_NAME=$(basename "$log" .log)
                echo -e "${CYAN}$SYSTEM_NAME:${NC} $LATEST_BALANCE"
            fi
        done
        
        echo ""
        echo -e "${YELLOW}🎯 Recent Milestones:${NC}"
        for log in $LOGS; do
            tail -50 "$log" | grep -E "(MILESTONE|TARGET.*ACHIEVED|MILLIONAIRE)" | tail -3
        done
        
    else
        echo -e "${YELLOW}⚠️  No active trading sessions found.${NC}"
    fi
}

# Function to show profit breakdown
show_profit_breakdown() {
    echo ""
    echo -e "${GREEN}💰 PROFIT BREAKDOWN BY SYSTEM:${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
    
    LOGS=$(ls -t logs/*.log 2>/dev/null)
    
    for log in $LOGS; do
        echo -e "${CYAN}$(basename "$log" .log):${NC}"
        tail -100 "$log" | grep -E "(PROFIT|WIN)" | tail -5 | while read line; do
            echo "  $line"
        done
        echo ""
    done
}

# Function to show system status
show_system_status() {
    echo -e "${GREEN}⚡ SYSTEM STATUS:${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
    
    # Check if systems are running
    if pgrep -f "mega_orchestrator.py" > /dev/null; then
        echo -e "${GREEN}✅ Mega Orchestrator: RUNNING${NC}"
    else
        echo -e "${RED}❌ Mega Orchestrator: STOPPED${NC}"
    fi
    
    if pgrep -f "ultimate_coordinator.py" > /dev/null; then
        echo -e "${GREEN}✅ Ultimate Coordinator: RUNNING${NC}"
    else
        echo -e "${RED}❌ Ultimate Coordinator: STOPPED${NC}"
    fi
    
    if pgrep -f "ultimate_success_coordinator.py" > /dev/null; then
        echo -e "${GREEN}✅ Ultimate Success Coordinator: RUNNING${NC}"
    else
        echo -e "${RED}❌ Ultimate Success Coordinator: STOPPED${NC}"
    fi
    
    if pgrep -f "ai_auto_optimizer.py" > /dev/null; then
        echo -e "${GREEN}✅ AI Auto-Optimizer: RUNNING${NC}"
    else
        echo -e "${RED}❌ AI Auto-Optimizer: STOPPED${NC}"
    fi
}

# Main dashboard loop
main_dashboard() {
    while true; do
        show_header
        show_live_performance
        show_profit_breakdown
        show_system_status
        
        echo ""
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}Commands: [r]efresh [q]uit [s]tart [k]ill${NC}"
        echo -e "${GREEN}Auto-refresh in 10 seconds...${NC}"
        
        # Wait for input or auto-refresh
        read -t 10 -n 1 KEY
        
        case $KEY in
            'q'|'Q')
                echo -e "${GREEN}Dashboard closed.${NC}"
                exit 0
                ;;
            's'|'S')
                echo -e "${BLUE}Starting Ultimate Success System...${NC}"
                ./launch_ultimate_success.sh &
                sleep 2
                ;;
            'k'|'K')
                echo -e "${RED}Stopping all systems...${NC}"
                pkill -f "mega_orchestrator.py"
                pkill -f "ultimate_coordinator.py"
                pkill -f "ultimate_success_coordinator.py"
                pkill -f "ai_auto_optimizer.py"
                sleep 2
                ;;
            'r'|'R'|'')
                # Refresh (do nothing, loop will continue)
                ;;
        esac
    done
}

# Run the dashboard
main_dashboard
