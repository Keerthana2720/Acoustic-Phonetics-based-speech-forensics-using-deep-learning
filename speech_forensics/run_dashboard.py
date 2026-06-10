"""
Simple script to run the Streamlit dashboard
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "src", "app", "dashboard.py")
    
    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    subprocess.run(cmd)

