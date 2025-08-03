"""
Wrapper module for profit optimization to maintain compatibility with the original codebase.
"""
import os
import argparse
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.optimization.profit_optimizer import run_profit_optimization

def main():
    """
    Parse arguments and run profit optimization
    """
    parser = argparse.ArgumentParser(description='Profit optimization tool')
    args = parser.parse_args()
    
    run_profit_optimization()

if __name__ == "__main__":
    main()