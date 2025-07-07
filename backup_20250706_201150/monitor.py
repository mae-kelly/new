#!/usr/bin/env python3
"""
Trading bot monitoring dashboard
"""

import sqlite3
import json
from datetime import datetime, timedelta

def show_performance():
    """Show bot performance metrics"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        print("=" * 60)
        print("📊 TRADING BOT PERFORMANCE DASHBOARD")
        print("=" * 60)
        
        # Get all trades
        cursor.execute('''
            SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20
        ''')
        trades = cursor.fetchall()
        
        if not trades:
            print("📝 No trades recorded yet")
            return
        
        # Calculate metrics
        total_profit = sum(trade[9] for trade in trades if trade[9])  # profit_loss column
        win_count = len([t for t in trades if t[9] and t[9] > 0])
        total_trades = len(trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        print(f"💰 Total Profit/Loss: ${total_profit:.2f}")
        print(f"🎯 Win Rate: {win_rate:.1f}% ({win_count}/{total_trades})")
        print(f"📈 Total Trades: {total_trades}")
        print()
        
        print("📋 RECENT TRADES:")
        print("-" * 60)
        for trade in trades[:10]:
            timestamp = trade[1][:16]  # First 16 chars of timestamp
            token = trade[2][:8] + "..."  # First 8 chars of address
            action = trade[3]
            amount = trade[4] or 0
            price = trade[5] or 0
            profit = trade[9] or 0
            
            profit_str = f"${profit:+.2f}" if profit != 0 else "N/A"
            print(f"{timestamp} | {token} | {action:4} | {profit_str:>8}")
        
        # Current positions
        cursor.execute('SELECT * FROM positions')
        positions = cursor.fetchall()
        
        if positions:
            print()
            print("📍 ACTIVE POSITIONS:")
            print("-" * 60)
            for pos in positions:
                token = pos[0][:8] + "..."
                entry_price = pos[1]
                amount = pos[2]
                entry_time = pos[3][:16]
                print(f"{token} | Entry: ${entry_price:.6f} | Time: {entry_time}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_performance()
