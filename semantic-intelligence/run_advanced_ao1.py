#!/usr/bin/env python3

import sys
import os

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Combine the two engine parts
from advanced_ao1_engine_part1 import AdvancedAO1Engine
from advanced_ao1_engine_part2 import AdvancedAO1EngineExecutor

# Monkey patch the methods from part 2 into the main engine
def combine_engine_parts():
    executor = AdvancedAO1EngineExecutor(None)  # We'll set engine later
    
    # Copy all executor methods to the main engine class
    for method_name in dir(executor):
        if method_name.startswith('_build_') or method_name.startswith('_extract_') or method_name.startswith('_analyze_') or method_name.startswith('_generate_') or method_name.startswith('_get_') or method_name.startswith('_calculate_dashboard'):
            if not hasattr(AdvancedAO1Engine, method_name):
                method = getattr(executor, method_name)
                # Create a wrapper that sets self properly
                def make_wrapper(original_method):
                    def wrapper(self, *args, **kwargs):
                        # Set the engine reference for the executor
                        original_method.__self__.engine = self
                        return original_method(*args, **kwargs)
                    return wrapper
                
                setattr(AdvancedAO1Engine, method_name, make_wrapper(method))

# Apply the combination
combine_engine_parts()

if __name__ == "__main__":
    print("🧠 Advanced AO1 Engine Combined Successfully!")
    print("Ready to run intelligent semantic analysis.")
    
    # Example usage
    if len(sys.argv) > 1:
        from main import main
        main()
    else:
        print("\nTo run analysis:")
        print("python run_advanced_ao1.py -d your_database.sqlite --intelligent-ao1")
        print("python run_advanced_ao1.py -d your_database.duckdb --ao1-advanced --save-results")