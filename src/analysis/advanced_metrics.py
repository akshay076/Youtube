"""
Advanced metrics module for Delhi Capitals player analysis.
Calculates sophisticated performance indicators and predictive metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMetricsAnalyzer:
    """Class to calculate advanced cricket metrics for breakout player prediction."""
    
    def __init__(self, player_data, basic_metrics=None):
        """
        Initialize the advanced metrics analyzer with player data.
        
        Args:
            player_data (dict): Player statistics and match data
            basic_metrics (dict, optional): Pre-calculated basic metrics
        """
        self.player_data = player_data
        self.player_name = player_data.get("player_name", "Unknown Player")
        self.match_data = player_data.get("match_data", [])
        
        # Load basic metrics if provided, otherwise attempt to load from file
        if basic_metrics:
            self.basic_metrics = basic_metrics
        else:
            metrics_file = os.path.join(METRICS_DIR, f"{self.player_name.replace(' ', '_')}_basic_metrics.json")
            try:
                with open(metrics_file, 'r') as f:
                    self.basic_metrics = json.load(f)
                logger.info(f"Loaded basic metrics for {self.player_name} from file")
            except FileNotFoundError:
                logger.warning(f"Basic metrics file for {self.player_name} not found")
                self.basic_metrics = {}
        
        self.advanced_metrics = {}
        logger.info(f"AdvancedMetricsAnalyzer initialized for {self.player_name}")
    
    def calculate_form_metrics(self):
        """
        Calculate form-based metrics using recent performance data.
        
        Returns:
            dict: Dictionary of form metrics
        """
        logger.info(f"Calculating form metrics for {self.player_name}")
        
        if not self.match_data:
            logger.warning(f"No match data available for {self.player_name}")
            return {}
        
        # Sort match data by date (most recent first)
        sorted_matches = sorted(self.match_data, key=lambda x: x.get("date", ""), reverse=True)
        
        # Calculate recent form (last 5 matches or all if less than 5)
        recent_matches = sorted_matches[:min(5, len(sorted_matches))]
        recent_runs = [m.get("runs", 0) for m in recent_matches]
        recent_balls = [m.get("balls_faced", 0) for m in recent_matches if m.get("balls_faced", 0) > 0]
        recent_strike_rates = [m.get("strike_rate", 0) for m in recent_matches if m.get("strike_rate", 0) > 0]
        
        recent_avg = sum(recent_runs) / len(recent_matches) if recent_matches else 0
        recent_sr = sum(m.get("runs", 0) for m in recent_matches) / sum(m.get("balls_faced", 0) for m in recent_matches) * 100 if sum(m.get("balls_faced", 0) for m in recent_matches) > 0 else 0
        
        # Calculate career average for comparison
        career_avg = self.player_data.get("average", 0)
        career_sr = self.player_data.get("strike_rate", 0)
        
        # Form index: ratio of recent average to career average
        form_index = (recent_avg / career_avg * 100) if career_avg > 0 else 0
        
        # Strike rate trend
        sr_trend = (recent_sr / career_sr * 100) if career_sr > 0 else 0
        
        # Calculate consistency (coefficient of variation - lower is more consistent)
        cv = np.std(recent_runs) / np.mean(recent_runs) if np.mean(recent_runs) > 0 else float('inf')
        consistency_score = 100 - min(100, cv * 50)  # Transform to 0-100 scale where higher is better
        
        # Calculate performance trajectory using linear regression
        if len(sorted_matches) >= 3:
            # Convert dates to numeric values (days since first match)
            try:
                dates = [datetime.strptime(m.get("date", "2024-01-01"), "%Y-%m-%d") for m in sorted_matches]
                days_since_first = [(d - dates[-1]).days for d in dates]
                
                # Extract performance metrics for regression
                performance_values = [m.get("runs", 0) for m in sorted_matches]
                
                # Reshape for sklearn
                X = np.array(days_since_first).reshape(-1, 1)
                y = np.array(performance_values)
                
                # Fit regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Get slope coefficient
                trajectory_slope = model.coef_[0]
                
                # Normalize slope to create trajectory score (0-100)
                # Positive slope indicates improvement
                max_expected_slope = 2  # Maximum expected improvement per day
                trajectory_score = min(100, max(0, (trajectory_slope / max_expected_slope) * 50 + 50))
            except Exception as e:
                logger.error(f"Error calculating trajectory for {self.player_name}: {e}")
                trajectory_score = 50  # Neutral score on error
        else:
            trajectory_score = 50  # Neutral score if not enough matches
        
        # Form metrics dictionary
        form_metrics = {
            "recent_average": round(recent_avg, 2),
            "recent_strike_rate": round(recent_sr, 2),
            "form_index": round(form_index, 2),
            "strike_rate_trend": round(sr_trend, 2),
            "consistency_score": round(consistency_score, 2),
            "trajectory_score": round(trajectory_score, 2),
            "recent_matches_analyzed": len(recent_matches)
        }
        
        logger.info(f"Completed form metrics calculation for {self.player_name}")
        return form_metrics
    
    def calculate_opposition_quality_metrics(self):
        """
        Calculate metrics that adjust for opposition quality.
        
        Returns:
            dict: Dictionary of opposition-adjusted metrics
        """
        logger.info(f"Calculating opposition quality metrics for {self.player_name}")
        
        if not self.match_data:
            logger.warning(f"No match data available for {self.player_name}")
            return {}
        
        # Extract opposition strength and performance data
        opposition_data = [(m.get("opposition", "Unknown"), m.get("opposition_strength", 5), m.get("runs", 0)) 
                          for m in self.match_data]
        
        # Calculate average performance against different opposition strengths
        strength_bins = {
            "weak": [1, 2, 3, 4],
            "average": [5, 6, 7],
            "strong": [8, 9, 10]
        }
        
        performance_by_strength = {category: [] for category in strength_bins}
        
        for _, strength, runs in opposition_data:
            for category, values in strength_bins.items():
                if strength in values:
                    performance_by_strength[category].append(runs)
        
        # Calculate average performance against each strength category
        avg_vs_weak = np.mean(performance_by_strength["weak"]) if performance_by_strength["weak"] else 0
        avg_vs_average = np.mean(performance_by_strength["average"]) if performance_by_strength["average"] else 0
        avg_vs_strong = np.mean(performance_by_strength["strong"]) if performance_by_strength["strong"] else 0
        
        # Calculate overall average for comparison
        overall_avg = self.player_data.get("average", 0)
        
        # Calculate opposition quality adjustment factor
        # Higher score if performing well against strong opposition
        strong_performance_ratio = avg_vs_strong / overall_avg if overall_avg > 0 else 0
        opposition_quality_score = min(100, strong_performance_ratio * 50)
        
        # Calculate performance against specific teams
        teams = {}
        for opposition, _, runs in opposition_data:
            if opposition not in teams:
                teams[opposition] = {"runs": [], "matches": 0}
            teams[opposition]["runs"].append(runs)
            teams[opposition]["matches"] += 1
        
        team_averages = {team: np.mean(data["runs"]) for team, data in teams.items()}
        
        # Calculate big match temperament (performance in high-pressure situations)
        high_pressure_matches = [m for m in self.match_data if m.get("high_pressure", False)]
        high_pressure_avg = np.mean([m.get("runs", 0) for m in high_pressure_matches]) if high_pressure_matches else 0
        big_match_temperament = (high_pressure_avg / overall_avg * 100) if overall_avg > 0 else 0
        
        # Opposition quality metrics dictionary
        opposition_metrics = {
            "avg_vs_weak_opposition": round(avg_vs_weak, 2),
            "avg_vs_average_opposition": round(avg_vs_average, 2),
            "avg_vs_strong_opposition": round(avg_vs_strong, 2),
            "opposition_quality_score": round(opposition_quality_score, 2),
            "big_match_temperament": round(big_match_temperament, 2),
            "team_specific_averages": {team: round(avg, 2) for team, avg in team_averages.items()}
        }
        
        logger.info(f"Completed opposition quality metrics calculation for {self.player_name}")
        return opposition_metrics
    
    def calculate_venue_condition_metrics(self):
        """
        Calculate metrics related to venue and match conditions.
        
        Returns:
            dict: Dictionary of venue and condition metrics
        """
        logger.info(f"Calculating venue and condition metrics for {self.player_name}")
        
        if not self.match_data:
            logger.warning(f"No match data available for {self.player_name}")
            return {}
        
        # Extract home/away status
        home_matches = [m for m in self.match_data if m.get("home_game", False)]
        away_matches = [m for m in self.match_data if not m.get("home_game", False)]
        
        # Calculate performance differences
        home_avg = np.mean([m.get("runs", 0) for m in home_matches]) if home_matches else 0
        away_avg = np.mean([m.get("runs", 0) for m in away_matches]) if away_matches else 0
        
        # Home advantage factor
        home_advantage_factor = (home_avg / away_avg) if away_avg > 0 else 1
        
        # Venue adaptability score (higher if performs well both home and away)
        venue_adaptability = 100 - min(100, abs(home_avg - away_avg) * 2)
        
        # Venue and condition metrics dictionary
        venue_metrics = {
            "home_average": round(home_avg, 2),
            "away_average": round(away_avg, 2),
            "home_matches": len(home_matches),
            "away_matches": len(away_matches),
            "home_advantage_factor": round(home_advantage_factor, 2),
            "venue_adaptability_score": round(venue_adaptability, 2)
        }
        
        logger.info(f"Completed venue and condition metrics calculation for {self.player_name}")
        return venue_metrics
    
    def calculate_impact_metrics(self):
        """
        Calculate impact-based metrics that measure player's contribution to team success.
        
        Returns:
            dict: Dictionary of impact metrics
        """
        logger.info(f"Calculating impact metrics for {self.player_name}")
        
        # Extract batting metrics
        batting_avg = self.player_data.get("average", 0)
        strike_rate = self.player_data.get("strike_rate", 0)
        
        # Calculate Batting Impact Index (Strike Rate × Average / 100)
        batting_impact_index = (strike_rate * batting_avg / 100) if batting_avg > 0 else 0
        
        # Calculate Phase Impact Score based on strike rates in different phases
        phase_metrics = self.basic_metrics.get("phase", {})
        powerplay_sr = phase_metrics.get("powerplay_strike_rate", 0)
        middle_sr = phase_metrics.get("middle_strike_rate", 0)
        death_sr = phase_metrics.get("death_strike_rate", 0)
        
        # Weights for different phases (adjust based on importance)
        powerplay_weight = 0.3
        middle_weight = 0.3
        death_weight = 0.4
        
        # Calculate weighted phase impact
        phase_impact_score = (
            (powerplay_sr * powerplay_weight) +
            (middle_sr * middle_weight) +
            (death_sr * death_weight)
        ) / 100  # Normalize to 0-1 scale
        
        # Boundary Impact Score (% of runs from boundaries)
        total_runs = self.player_data.get("total_runs", 0)
        total_fours = self.player_data.get("total_fours", 0)
        total_sixes = self.player_data.get("total_sixes", 0)
        
        boundary_runs = (total_fours * 4) + (total_sixes * 6)
        boundary_impact = (boundary_runs / total_runs) if total_runs > 0 else 0
        
        # Bowling impact (if applicable)
        bowling_impact_index = 0
        if "Bowler" in self.player_data.get("role", "") or "All-rounder" in self.player_data.get("role", ""):
            economy = self.player_data.get("economy_rate", 0)
            bowling_strike_rate = self.player_data.get("bowling_strike_rate", 0)
            bowling_avg = self.player_data.get("bowling_average", 0)
            
            # Calculate bowling impact (lower is better for economy and average)
            if bowling_strike_rate > 0 and economy > 0:
                # Transform so higher is better
                bowling_impact_index = 100 / (bowling_strike_rate * economy / 20)
        
        # Calculate overall player impact score (normalized to 0-100)
        overall_impact = 0
        role = self.player_data.get("role", "")
        
        if "Batsman" in role or "WK" in role:
            # Primarily batting impact
            overall_impact = min(100, batting_impact_index * 2)
        elif "Bowler" in role:
            # Primarily bowling impact
            overall_impact = min(100, bowling_impact_index * 2)
        elif "All-rounder" in role:
            # Combined impact
            overall_impact = min(100, (batting_impact_index + bowling_impact_index) * 1.25)
        
        # Impact metrics dictionary
        impact_metrics = {
            "batting_impact_index": round(batting_impact_index, 2),
            "phase_impact_score": round(phase_impact_score * 100, 2),  # Convert to 0-100 scale
            "boundary_impact_score": round(boundary_impact * 100, 2),  # Convert to 0-100 scale
            "bowling_impact_index": round(bowling_impact_index, 2),
            "overall_impact_score": round(overall_impact, 2)
        }
        
        logger.info(f"Completed impact metrics calculation for {self.player_name}")
        return impact_metrics
    
    def calculate_breakout_potential(self):
        """
        Calculate breakout potential score based on all metrics.
        
        Returns:
            dict: Dictionary with breakout potential metrics
        """
        logger.info(f"Calculating breakout potential for {self.player_name}")
        
        # Get metrics from different categories
        form_metrics = self.advanced_metrics.get("form", {})
        opposition_metrics = self.advanced_metrics.get("opposition_quality", {})
        venue_metrics = self.advanced_metrics.get("venue_condition", {})
        impact_metrics = self.advanced_metrics.get("impact", {})
        
        # Extract key indicators
        form_index = form_metrics.get("form_index", 0)
        trajectory_score = form_metrics.get("trajectory_score", 0)
        opposition_quality_score = opposition_metrics.get("opposition_quality_score", 0)
        big_match_temperament = opposition_metrics.get("big_match_temperament", 0)
        venue_adaptability = venue_metrics.get("venue_adaptability_score", 0)
        overall_impact = impact_metrics.get("overall_impact_score", 0)
        
        # Calculate weighted breakout potential score
        breakout_score = (
            (form_index * 0.15) +
            (trajectory_score * 0.25) +
            (opposition_quality_score * 0.15) +
            (big_match_temperament * 0.15) +
            (venue_adaptability * 0.1) +
            (overall_impact * 0.2)
        )
        
        # Normalize to 0-100 scale
        breakout_score = min(100, max(0, breakout_score * 0.8))
        
        # Breakout potential category
        if breakout_score >= 80:
            category = "Very High"
        elif breakout_score >= 65:
            category = "High"
        elif breakout_score >= 50:
            category = "Moderate"
        elif breakout_score >= 35:
            category = "Low"
        else:
            category = "Very Low"
        
        # Key strengths (top 3 metrics)
        metrics_list = [
            ("Form", form_index),
            ("Trajectory", trajectory_score),
            ("Opposition Quality", opposition_quality_score),
            ("Big Match Temperament", big_match_temperament),
            ("Venue Adaptability", venue_adaptability),
            ("Overall Impact", overall_impact)
        ]
        
        # Sort by value (descending) and get top 3
        metrics_list.sort(key=lambda x: x[1], reverse=True)
        key_strengths = [metric[0] for metric in metrics_list[:3]]
        
        # Areas for improvement (bottom 2 metrics)
        areas_for_improvement = [metric[0] for metric in metrics_list[-2:]]
        
        # Breakout potential metrics dictionary
        breakout_metrics = {
            "breakout_potential_score": round(breakout_score, 2),
            "breakout_category": category,
            "key_strengths": key_strengths,
            "areas_for_improvement": areas_for_improvement,
            "experience_level": "Low" if self.player_data.get("matches", 0) <= 10 else "Moderate"
        }
        
        logger.info(f"Completed breakout potential calculation for {self.player_name}")
        return breakout_metrics
    
    def calculate_all_metrics(self):
        """
        Calculate all advanced performance metrics.
        
        Returns:
            dict: Dictionary of all advanced metrics
        """
        logger.info(f"Calculating all advanced metrics for {self.player_name}")
        
        # Basic player info
        self.advanced_metrics["player_name"] = self.player_name
        self.advanced_metrics["role"] = self.player_data.get("role", "Unknown")
        self.advanced_metrics["matches"] = self.player_data.get("matches", 0)
        
        # Calculate specific metric categories
        self.advanced_metrics["form"] = self.calculate_form_metrics()
        self.advanced_metrics["opposition_quality"] = self.calculate_opposition_quality_metrics()
        self.advanced_metrics["venue_condition"] = self.calculate_venue_condition_metrics()
        self.advanced_metrics["impact"] = self.calculate_impact_metrics()
        
        # Calculate breakout potential (must be done after other metrics)
        self.advanced_metrics["breakout_potential"] = self.calculate_breakout_potential()
        
        # Save the metrics
        self.save_metrics()
        
        logger.info(f"Completed all advanced metrics calculation for {self.player_name}")
        return self.advanced_metrics
    
    def save_metrics(self):
        """Save the calculated metrics to a JSON file."""
        metrics_file = os.path.join(METRICS_DIR, f"{self.player_name.replace(' ', '_')}_advanced_metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.advanced_metrics, f, indent=4)
        
        logger.info(f"Saved advanced metrics for {self.player_name} to {metrics_file}")

# For testing
if __name__ == "__main__":
    # Test with sample player data
    from src.data.data_collector import DataCollector
    
    # Initialize data collector
    collector = DataCollector()
    
    # Get data for a test player
    test_player = POTENTIAL_BREAKOUT_PLAYERS[0]  # First potential breakout player
    player_data = collector.get_player_stats(test_player)
    
    # Initialize metrics analyzers
    from src.analysis.basic_metrics import BasicMetricsAnalyzer
    basic_analyzer = BasicMetricsAnalyzer(player_data)
    basic_metrics = basic_analyzer.calculate_all_metrics()
    
    # Initialize advanced metrics analyzer with basic metrics
    advanced_analyzer = AdvancedMetricsAnalyzer(player_data, basic_metrics)
    
    # Calculate all advanced metrics
    advanced_metrics = advanced_analyzer.calculate_all_metrics()
    
    # Print results
    print(f"Advanced metrics for {test_player['name']}:")
    print(json.dumps(advanced_metrics, indent=2))