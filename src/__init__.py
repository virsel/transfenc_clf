import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current file's directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the sys.path
sys.path.append(parent_dir)