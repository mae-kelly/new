#!/usr/bin/env python3
"""
Database initialization for Ultimate Trading System
"""
import sqlite3
import os
from datetime import datetime

def init_database():
    """Initialize the trading system database"""
    
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect('data/ultimate_trading.db')
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                phase INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL NOT NULL,
                balance_after REAL NOT NULL,
                confidence_score REAL
            )
        ''')
        
        # Create portfolio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                phase INTEGER NOT NULL,
                total_balance REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                total_return REAL NOT NULL
            )
        ''')
        
        # Create system events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT DEFAULT 'INFO'
            )
        ''')
        
        # Insert initial event
        cursor.execute('''
            INSERT INTO system_events (timestamp, event_type, description)
            VALUES (?, 'SYSTEM_INIT', 'Database initialized successfully')
        ''', (datetime.now().isoformat(),))
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    init_database()
