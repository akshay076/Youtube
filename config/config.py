"""
Configuration settings for Delhi Capitals analysis project
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, OUTPUT_DIR, METRICS_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# API configuration
CRICINFO_BASE_URL = "https://www.espncricinfo.com"
IPLT20_BASE_URL = "https://www.iplt20.com"
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Match phases (in overs)
POWERPLAY_OVERS = range(1, 7)
MIDDLE_OVERS = range(7, 16)
DEATH_OVERS = range(16, 21)

# Performance thresholds
BREAKOUT_BATTING_IMPACT_THRESHOLD = 40  # Batting Impact Index threshold for potential breakout
HIGH_STRIKE_RATE_THRESHOLD = 150        # Strike rate threshold for aggressive batting
CONSISTENCY_THRESHOLD = 70              # Consistency Index threshold for reliable performance

# Analysis weights
FORM_WEIGHT = 0.3                       # Weight for recent form
OPPOSITION_QUALITY_WEIGHT = 0.25        # Weight for opposition quality adjustment
CONSISTENCY_WEIGHT = 0.2                # Weight for consistency
IMPACT_WEIGHT = 0.25                    # Weight for match impact

# Delhi Capitals full squad for IPL 2025 (as example)
# This would be ideally fetched from a database or API
DELHI_CAPITALS_SQUAD = [
    # Key players
    {"name": "Rishabh Pant", "role": "WK-Batsman", "experience": 98},
    {"name": "Axar Patel", "role": "All-rounder", "experience": 125},
    {"name": "Kuldeep Yadav", "role": "Bowler", "experience": 82},
    {"name": "Tristan Stubbs", "role": "Batsman", "experience": 16},
    {"name": "Jake Fraser-McGurk", "role": "Batsman", "experience": 9},
    
    # Other squad members
    {"name": "Prithvi Shaw", "role": "Batsman", "experience": 64},
    {"name": "Mitchell Marsh", "role": "All-rounder", "experience": 48},
    {"name": "Abishek Porel", "role": "WK-Batsman", "experience": 14},
    {"name": "Anrich Nortje", "role": "Bowler", "experience": 43},
    {"name": "Mukesh Kumar", "role": "Bowler", "experience": 18},
    {"name": "Khaleel Ahmed", "role": "Bowler", "experience": 37},
    {"name": "Ishant Sharma", "role": "Bowler", "experience": 93},
    {"name": "Jhye Richardson", "role": "Bowler", "experience": 8},
    {"name": "Rasikh Dar", "role": "Bowler", "experience": 5},
    {"name": "Sameer Rizvi", "role": "Batsman", "experience": 4},
    {"name": "Sumit Kumar", "role": "All-rounder", "experience": 3},
    {"name": "Kumar Kushagra", "role": "WK-Batsman", "experience": 2},
    {"name": "Shai Hope", "role": "WK-Batsman", "experience": 7},
    {"name": "Ashutosh Sharma", "role": "All-rounder", "experience": 6},
    {"name": "Ricky Bhui", "role": "Batsman", "experience": 5},
    {"name": "Harry Brook", "role": "Batsman", "experience": 15},
    {"name": "Vicky Ostwal", "role": "Bowler", "experience": 2},
    {"name": "Swastik Chikara", "role": "Batsman", "experience": 0}
]

# Potential breakout players (≤20 IPL matches)
POTENTIAL_BREAKOUT_PLAYERS = [player for player in DELHI_CAPITALS_SQUAD if player["experience"] <= 20]