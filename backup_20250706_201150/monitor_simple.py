#!/usr/bin/env python3
"""
Simple trading bot monitor
"""

import sqlite3
from datetime import datetime

def show_results():
    try:
        conn = sqlite3.connect('simple_bot.db')
        cursor = conn.cursor()
        
        print("=" * 50)
        print("📊 TRADING BOT RESULTS")
        print("=" * 50)
        
        # Get all trades
        cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC')
        trades = cursor.fetchall()
        
        if not trades:
            print("📝 No trades found")
            return
        
        # Calculate performance
        buys = [t for t in trades if t[3] == 'BUY']
        sells = [t for t in trades if t[3] == 'SELL']
        
        total_profit = sum(t[6] for t in sells if t[6])
        total_trades = len(buys)
        wins = len([s for s in sells if s[6] and s[6] > 0])
        
        if trades:
            final_balance = trades[0][7]  # Most recent balance
            print(f"💰 Final Balance: ${final_balance:.2f}")
            
        print(f"📈 Total Profit: ${total_profit:.2f}")
        print(f"🎯 Win Rate: {wins}/{len(sells)} = {(wins/len(sells)*100) if sells else 0:.1f}%")
        print(f"📊 Total Trades: {total_trades}")
        print()
        
        print("📋 RECENT TRADES:")
        print("-" * 50)
        for trade in trades[:10]:
            timestamp = trade[1][:16]
            token = trade[2][:8] + "..."
            action = trade[3]
            amount = trade[4]
            profit = trade[6] if trade[6] else 0
            
            print(f"{timestamp} | {action:4} | ${amount:6.2f} | P&L: ${profit:+6.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_results()
