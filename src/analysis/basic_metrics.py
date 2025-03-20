"""
Basic metrics module for Delhi Capitals player analysis.
Calculates fundamental cricket statistics and performance indicators.
"""

import os
import json
import pandas as pd
import numpy as np
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicMetricsAnalyzer:
    """Class to calculate basic cricket metrics for player analysis."""
    
    def __init__(self, player_data):
        """
        Initialize the basic metrics analyzer with player data.
        
        Args:
            player_data (dict): Player statistics and match data
        """
        self.player_data = player_data
        self.player_name = player_data.get("player_name", "Unknown Player")
        self.metrics = {}
        logger.info(f"Calculating bowling metrics for {self.player_name}")
        
        # Only calculate bowling metrics if player has bowling data
        if "Bowler" not in self.player_data.get("role", "") and "All-rounder" not in self.player_data.get("role", ""):
            logger.info(f"{self.player_name} is not a bowler - skipping bowling metrics")
            return {}
        
        # Extract bowling stats
        matches = self.player_data.get("matches", 0)
        bowling_innings = self.player_data.get("bowling_innings", 0)
        total_wickets = self.player_data.get("total_wickets", 0)
        economy_rate = self.player_data.get("economy_rate", 0)
        bowling_strike_rate = self.player_data.get("bowling_strike_rate", 0)
        bowling_average = self.player_data.get("bowling_average", 0)
        
        # Calculate additional metrics
        wickets_per_match = total_wickets / matches if matches > 0 else 0
        
        # Calculate variations across phases (powerplay, middle, death)
        match_data = self.player_data.get("match_data", [])
        
        # In a real implementation, we would have detailed bowling phase data
        # For this demo, we'll use simulated data
        
        # Store metrics in dictionary
        bowling_metrics = {
            "bowling_innings": bowling_innings,
            "total_wickets": total_wickets,
            "economy_rate": round(economy_rate, 2),
            "bowling_strike_rate": round(bowling_strike_rate, 2),
            "bowling_average": round(bowling_average, 2),
            "wickets_per_match": round(wickets_per_match, 2),
        }
        
        logger.info(f"Completed bowling metrics calculation for {self.player_name}")
        return bowling_metrics
    
    def calculate_fielding_metrics(self):
        """
        Calculate basic fielding performance metrics.
        
        Returns:
            dict: Dictionary of fielding metrics
        """
        logger.info(f"Calculating fielding metrics for {self.player_name}")
        
        # In a real implementation, we would extract actual fielding data
        # For this demo, we'll generate synthetic metrics
        match_data = self.player_data.get("match_data", [])
        matches = self.player_data.get("matches", 0)
        
        # Generate synthetic fielding data based on player role
        role = self.player_data.get("role", "")
        
        # Simulate catches and run-outs based on role and experience
        if "WK" in role:
            # Wicketkeepers tend to have more catches and stumpings
            catches = int(matches * 1.2)
            stumpings = int(matches * 0.3)
            run_outs = int(matches * 0.1)
        elif "Bowler" in role:
            # Bowlers get some catches off their own bowling
            catches = int(matches * 0.6)
            stumpings = 0
            run_outs = int(matches * 0.15)
        elif "Batsman" in role:
            # Regular fielders
            catches = int(matches * 0.5)
            stumpings = 0
            run_outs = int(matches * 0.15)
        else:  # All-rounders
            catches = int(matches * 0.7)
            stumpings = 0
            run_outs = int(matches * 0.2)
        
        # Add some randomness
        catches = max(0, int(catches * np.random.normal(1, 0.2)))
        stumpings = max(0, int(stumpings * np.random.normal(1, 0.2)))
        run_outs = max(0, int(run_outs * np.random.normal(1, 0.2)))
        
        # Calculate fielding metrics
        dismissals = catches + stumpings + run_outs
        dismissals_per_match = dismissals / matches if matches > 0 else 0
        
        # Store metrics in dictionary
        fielding_metrics = {
            "catches": catches,
            "stumpings": stumpings,
            "run_outs": run_outs,
            "total_dismissals": dismissals,
            "dismissals_per_match": round(dismissals_per_match, 2)
        }
        
        logger.info(f"Completed fielding metrics calculation for {self.player_name}")
        return fielding_metrics
    
    def calculate_phase_metrics(self):
        """
        Calculate performance metrics across different match phases.
        
        Returns:
            dict: Dictionary of phase-specific metrics
        """
        logger.info(f"Calculating phase metrics for {self.player_name}")
        
        match_data = self.player_data.get("match_data", [])
        
        # Calculate batting metrics by phase
        total_powerplay_runs = sum(m.get("powerplay_runs", 0) for m in match_data)
        total_middle_runs = sum(m.get("middle_runs", 0) for m in match_data)
        total_death_runs = sum(m.get("death_runs", 0) for m in match_data)
        
        # In a real implementation, we would have detailed ball-by-ball data
        # For this demo, we'll estimate balls faced in each phase
        total_runs = self.player_data.get("total_runs", 0)
        total_balls = self.player_data.get("total_balls_faced", 0)
        
        # Estimate balls faced in each phase based on run distribution
        if total_runs > 0:
            powerplay_balls = int(total_balls * (total_powerplay_runs / total_runs))
            middle_balls = int(total_balls * (total_middle_runs / total_runs))
            death_balls = total_balls - powerplay_balls - middle_balls
        else:
            powerplay_balls = middle_balls = death_balls = 0
        
        # Calculate strike rates by phase
        powerplay_sr = (total_powerplay_runs / powerplay_balls * 100) if powerplay_balls > 0 else 0
        middle_sr = (total_middle_runs / middle_balls * 100) if middle_balls > 0 else 0
        death_sr = (total_death_runs / death_balls * 100) if death_balls > 0 else 0
        
        # Store metrics in dictionary
        phase_metrics = {
            "powerplay_runs": total_powerplay_runs,
            "powerplay_balls": powerplay_balls,
            "powerplay_strike_rate": round(powerplay_sr, 2),
            "powerplay_percentage": round((total_powerplay_runs / total_runs * 100) if total_runs > 0 else 0, 2),
            
            "middle_runs": total_middle_runs,
            "middle_balls": middle_balls,
            "middle_strike_rate": round(middle_sr, 2),
            "middle_percentage": round((total_middle_runs / total_runs * 100) if total_runs > 0 else 0, 2),
            
            "death_runs": total_death_runs,
            "death_balls": death_balls,
            "death_strike_rate": round(death_sr, 2),
            "death_percentage": round((total_death_runs / total_runs * 100) if total_runs > 0 else 0, 2)
        }
        
        logger.info(f"Completed phase metrics calculation for {self.player_name}")
        return phase_metrics
    
    def calculate_all_metrics(self):
        """
        Calculate all basic performance metrics.
        
        Returns:
            dict: Dictionary of all basic metrics
        """
        logger.info(f"Calculating all basic metrics for {self.player_name}")
        
        # Basic player info
        self.metrics["player_name"] = self.player_name
        self.metrics["role"] = self.player_data.get("role", "Unknown")
        self.metrics["matches"] = self.player_data.get("matches", 0)
        
        # Calculate specific metric categories
        self.metrics["batting"] = self.calculate_batting_metrics()
        self.metrics["bowling"] = self.calculate_bowling_metrics()
        self.metrics["fielding"] = self.calculate_fielding_metrics()
        self.metrics["phase"] = self.calculate_phase_metrics()
        
        # Save the metrics
        self.save_metrics()
        
        logger.info(f"Completed all basic metrics calculation for {self.player_name}")
        return self.metrics
    
    def save_metrics(self):
        """Save the calculated metrics to a JSON file."""
        metrics_file = os.path.join(METRICS_DIR, f"{self.player_name.replace(' ', '_')}_basic_metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Saved basic metrics for {self.player_name} to {metrics_file}")

# For testing
if __name__ == "__main__":
    # Test with sample player data
    from src.data.data_collector import DataCollector
    
    # Initialize data collector
    collector = DataCollector()
    
    # Get data for a test player
    test_player = DELHI_CAPITALS_SQUAD[0]  # First player in the squad
    player_data = collector.get_player_stats(test_player)
    
    # Initialize metrics analyzer
    analyzer = BasicMetricsAnalyzer(player_data)
    
    # Calculate all metrics
    metrics = analyzer.calculate_all_metrics()
    
    # Print results
    print(f"Basic metrics for {test_player['name']}:")
    print(json.dumps(metrics, indent=2))(f"BasicMetricsAnalyzer initialized for {self.player_name}")
    
    def calculate_batting_metrics(self):
        """
        Calculate basic batting performance metrics.
        
        Returns:
            dict: Dictionary of batting metrics
        """
        logger.info(f"Calculating batting metrics for {self.player_name}")
        
        # Extract basic stats from player data
        innings = self.player_data.get("innings", 0)
        total_runs = self.player_data.get("total_runs", 0)
        total_balls = self.player_data.get("total_balls_faced", 0)
        total_fours = self.player_data.get("total_fours", 0)
        total_sixes = self.player_data.get("total_sixes", 0)
        not_outs = self.player_data.get("not_outs", 0)
        
        # Calculate metrics
        batting_avg = self.player_data.get("average", 0)
        strike_rate = self.player_data.get("strike_rate", 0)
        boundary_percentage = self.player_data.get("boundary_percentage", 0)
        
        # Calculate runs per innings
        runs_per_innings = total_runs / innings if innings > 0 else 0
        
        # Calculate boundary frequency
        balls_per_boundary = total_balls / (total_fours + total_sixes) if (total_fours + total_sixes) > 0 else float('inf')
        
        # Calculate dot ball percentage from match data
        match_data = self.player_data.get("match_data", [])
        total_dot_balls = 0
        
        for match in match_data:
            # In a real implementation, we would have ball-by-ball data
            # For this demo, we'll estimate dot balls based on runs and boundaries
            runs = match.get("runs", 0)
            fours = match.get("fours", 0)
            sixes = match.get("sixes", 0)
            balls_faced = match.get("balls_faced", 0)
            
            scoring_balls = runs - (fours * 4) - (sixes * 6)
            # Assume each non-boundary scoring ball is a single
            singles_and_doubles = scoring_balls
            estimated_dot_balls = balls_faced - fours - sixes - singles_and_doubles
            total_dot_balls += max(0, estimated_dot_balls)  # Ensure non-negative
        
        dot_ball_percentage = (total_dot_balls / total_balls * 100) if total_balls > 0 else 0
        
        # Store metrics in dictionary
        batting_metrics = {
            "innings": innings,
            "total_runs": total_runs,
            "batting_average": round(batting_avg, 2),
            "strike_rate": round(strike_rate, 2),
            "boundary_percentage": round(boundary_percentage, 2),
            "runs_per_innings": round(runs_per_innings, 2),
            "balls_per_boundary": round(balls_per_boundary, 2) if balls_per_boundary != float('inf') else float('inf'),
            "dot_ball_percentage": round(dot_ball_percentage, 2),
            "highest_score": self.player_data.get("highest_score", 0),
            "fifty_plus": self.player_data.get("fifty_plus", 0),
            "hundred_plus": self.player_data.get("hundred_plus", 0),
            "ducks": self.player_data.get("ducks", 0),
            "not_outs": not_outs
        }
        
        logger.info(f"Completed batting metrics calculation for {self.player_name}")
        return batting_metrics
    
    def calculate_bowling_metrics(self):
        """
        Calculate basic bowling performance metrics.
        
        Returns:
            dict: Dictionary of bowling metrics
        """
        logger.info