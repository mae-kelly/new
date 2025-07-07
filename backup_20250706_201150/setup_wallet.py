#!/usr/bin/env python3
"""
Wallet setup script for Solana trading bot
"""

from solana.keypair import Keypair
import base58
import os

def create_new_wallet():
    """Create a new Solana wallet"""
    print("🔐 Creating new Solana wallet...")
    
    # Generate new keypair
    keypair = Keypair.generate()
    
    # Get private key in base58 format
    private_key = base58.b58encode(keypair.secret_key).decode('utf-8')
    public_key = str(keypair.public_key)
    
    print(f"✅ Wallet created!")
    print(f"📍 Public Key: {public_key}")
    print(f"🔑 Private Key: {private_key}")
    print()
    print("⚠️  IMPORTANT:")
    print("1. Save your private key securely")
    print("2. Add it to .env file: SOLANA_PRIVATE_KEY=your_private_key")
    print("3. Fund your wallet with SOL before live trading")
    print(f"4. You can view your wallet at: https://explorer.solana.com/address/{public_key}")
    
    # Update .env file
    env_content = ""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
    
    # Replace or add SOLANA_PRIVATE_KEY
    lines = env_content.split('\n')
    found = False
    for i, line in enumerate(lines):
        if line.startswith('SOLANA_PRIVATE_KEY='):
            lines[i] = f'SOLANA_PRIVATE_KEY={private_key}'
            found = True
            break
    
    if not found:
        lines.append(f'SOLANA_PRIVATE_KEY={private_key}')
    
    with open('.env', 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Private key added to .env file")

if __name__ == "__main__":
    create_new_wallet()
