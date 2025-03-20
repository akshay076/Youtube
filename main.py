"""
Delhi Capitals Breakout Player Analysis
Simplified execution script using real data only
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Import project modules
from config.config import *
from src.data.real_data_collector import RealDataCollector
from src.analysis.basic_metrics import BasicMetricsAnalyzer
from src.analysis.advanced_metrics import AdvancedMetricsAnalyzer
from src.visualization.player_cards import PlayerCardVisualizer
from src.visualization.comparison_plots import PlayerComparisonVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Delhi Capitals Breakout Player Analysis')
    parser.add_argument('--players', nargs='+', default=None,
                        help='Specific players to analyze (if not provided, all potential breakout players will be analyzed)')
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print(" DELHI CAPITALS BREAKOUT PLAYER ANALYSIS ".center(80, "="))
    print("="*80 + "\n")
    
    # Get players to analyze (if specified, otherwise use all potential breakout players)
    if args.players:
        # Find player info dictionaries for specified player names
        players_to_analyze = []
        for name in args.players:
            player_info = next((p for p in DELHI_CAPITALS_SQUAD if p["name"] == name), None)
            if player_info:
                players_to_analyze.append(player_info)
            else:
                logger.warning(f"Player {name} not found in Delhi Capitals squad")
    else:
        players_to_analyze = POTENTIAL_BREAKOUT_PLAYERS
    
    if not players_to_analyze:
        logger.error("No valid players to analyze. Exiting.")
        return
    
    print(f"Analyzing {len(players_to_analyze)} players for breakout potential...")
    
    # Initialize data collector
    collector = RealDataCollector()
    
    # Process each player
    results = []
    
    for player_info in players_to_analyze:
        player_name = player_info["name"]
        print(f"\nProcessing {player_name}...")
        
        # Step 1: Collect data
        print(f"  Collecting data...")
        player_data = collector.get_player_stats(player_info)
        
        if not player_data:
            logger.error(f"Could not collect data for {player_name}. Skipping.")
            continue
        
        # Save individual player data
        collector.save_player_data(player_name, player_data)
        
        # Step 2: Calculate basic metrics
        print(f"  Calculating basic metrics...")
        basic_analyzer = BasicMetricsAnalyzer(player_data)
        basic_metrics = basic_analyzer.calculate_all_metrics()
        
        # Step 3: Calculate advanced metrics
        print(f"  Calculating advanced metrics...")
        advanced_analyzer = AdvancedMetricsAnalyzer(player_data, basic_metrics)
        advanced_metrics = advanced_analyzer.calculate_all_metrics()
        
        # Step 4: Create player card
        print(f"  Creating visualization...")
        try:
            visualizer = PlayerCardVisualizer(player_name, basic_metrics, advanced_metrics)
            visualizer.create_player_card()
        except Exception as e:
            logger.error(f"Error creating player card for {player_name}: {e}")
        
        # Store breakout potential results
        breakout_potential = advanced_metrics.get("breakout_potential", {})
        results.append({
            "name": player_name,
            "role": player_data.get("role", "Unknown"),
            "score": breakout_potential.get("breakout_potential_score", 0),
            "category": breakout_potential.get("breakout_category", "N/A"),
            "key_strengths": breakout_potential.get("key_strengths", []),
            "areas_for_improvement": breakout_potential.get("areas_for_improvement", [])
        })
    
    # Save combined player data
    collector.save_combined_player_data({p["name"]: p for p in player_data})
    
    # Create comparison visualization for top players
    if len(results) > 1:
        print("\nCreating comparison visualization...")
        comparison_visualizer = PlayerComparisonVisualizer([r["name"] for r in results])
        try:
            comparison_visualizer.create_breakout_potential_comparison()
            comparison_visualizer.create_top_breakout_candidates(top_n=min(5, len(results)))
        except Exception as e:
            logger.error(f"Error creating comparison visualizations: {e}")
    
    # Print summary of results
    print("\n" + "="*80)
    print(" RESULTS SUMMARY ".center(80, "="))
    print("="*80)
    
    # Sort by breakout potential score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    for i, player in enumerate(results):
        print(f"{i+1}. {player['name']} ({player['role']})")
        print(f"   Breakout Potential: {player['score']:.1f} - {player['category']}")
        if player['key_strengths']:
            print(f"   Key Strengths: {', '.join(player['key_strengths'][:2])}")
        print()
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()