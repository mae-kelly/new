#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Setting up environment..."
    ./setup.sh
fi

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Could not find virtual environment"
    exit 1
fi

if [ ! -d "test_data" ]; then
    echo "Generating test data..."
    ./test.sh
fi

echo "Running Brilliant Analysis..."
python neural_main.py --mode test
