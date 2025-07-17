#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x run.sh
chmod +x test.sh
mkdir -p data models outputs logs cache
