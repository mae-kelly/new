#!/usr/bin/env python3
"""
Basic system monitoring
"""
import sqlite3
import os
from datetime import datetime

def get_system_status():
    """Get basic system status"""
    try:
        if not os.path.exists('data/ultimate_trading.db'):
            return "Database not found"
        
        conn = sqlite3.connect('data/ultimate_trading.db')
        cursor = conn.cursor()
        
        # Get latest portfolio data
        cursor.execute('SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT 1')
        latest = cursor.fetchone()
        
        # Get trade count
        cursor.execute('SELECT COUNT(*) FROM trades')
        trade_count = cursor.fetchone()[0]
        
        conn.close()
        
        if latest:
            balance = latest[3]
            total_return = latest[5]
            phase = latest[2]
        else:
            balance = 10.0
            total_return = 0.0
            phase = 1
        
        report = f"""
🚀 ULTIMATE TRADING SYSTEM STATUS
{'='*50}
💰 Current Balance: ${balance:.2f}
📈 Total Return: {total_return:+.2f}%
🏗️  Current Phase: {phase}
🔄 Total Trades: {trade_count}
⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}
        """
        
        return report
        
    except Exception as e:
        return f"❌ Monitoring error: {e}"

if __name__ == "__main__":
    print(get_system_status())
