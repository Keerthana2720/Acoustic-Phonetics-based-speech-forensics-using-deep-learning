#!/usr/bin/env python
"""
Training script wrapper for speech forensics model.
Run this script from the project root directory.
"""
import sys
import os

# Add the speech_forensics package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

if __name__ == '__main__':
    from speech_forensics.src import train
    train.main()
