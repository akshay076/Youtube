"""
Delhi Capitals Breakout Player Analysis
Main execution script
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Import project modules
from config.config import *
from src.data.improved_data_collector import ImprovedDataCollector
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Delhi Capitals Breakout Player Analysis')
    
    parser.add_argument('--collect', action='store_true', 
                        help='Collect player data')
    
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze player data')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    
    parser.add_argument('--all', action='store_true',
                        help='Run the complete pipeline')
    
    parser.add_argument('--players', nargs='+', default=None,
                        help='Specific players to analyze (if not provided, all potential breakout players will be analyzed)')
    
    return parser.parse_args()

def collect_data(players=None):
    """
    Collect player data.
    
    Args:
        players (list, optional): List of player info dictionaries.
                                  Defaults to potential breakout players.
    """
    logger.info("Starting data collection")
    
    # Initialize data collector
    collector = DataCollector()
    
    # Determine which players to collect data for
    if players is None:
        players_to_collect = POTENTIAL_BREAKOUT_PLAYERS
        logger.info(f"Collecting data for all {len(players_to_collect)} potential breakout players")
    else:
        # Find player info dictionaries for specified player names
        players_to_collect = []
        for name in players:
            player_info = next((p for p in DELHI_CAPITALS_SQUAD if p["name"] == name), None)
            if player_info:
                players_to_collect.append(player_info)
            else:
                logger.warning(f"Player {name} not found in Delhi Capitals squad")
        
        logger.info(f"Collecting data for {len(players_to_collect)} specified players")
    
    # Collect player data
    player_data = collector.collect_all_player_data(players_to_collect)
    
    logger.info(f"Data collection completed for {len(player_data)} players")
    return player_data

def analyze_data(players=None):
    """
    Analyze player data.
    
    Args:
        players (list, optional): List of player names.
                                  Defaults to potential breakout players.
    """
    logger.info("Starting data analysis")
    
    # Initialize data collector for loading data
    collector = DataCollector()
    
    # Determine which players to analyze
    if players is None:
        players_to_analyze = [p["name"] for p in POTENTIAL_BREAKOUT_PLAYERS]
        logger.info(f"Analyzing data for all {len(players_to_analyze)} potential breakout players")
    else:
        players_to_analyze = players
        logger.info(f"Analyzing data for {len(players_to_analyze)} specified players")
    
    # Analyze each player
    for player_name in players_to_analyze:
        logger.info(f"Analyzing player: {player_name}")
        
        # Load player data
        player_data = collector.load_player_data(player_name)
        if not player_data:
            logger.error(f"No data found for player {player_name}. Skipping analysis.")
            continue
        
        # Calculate basic metrics
        logger.info(f"Calculating basic metrics for {player_name}")
        basic_analyzer = BasicMetricsAnalyzer(player_data)
        basic_metrics = basic_analyzer.calculate_all_metrics()
        
        # Calculate advanced metrics
        logger.info(f"Calculating advanced metrics for {player_name}")
        advanced_analyzer = AdvancedMetricsAnalyzer(player_data, basic_metrics)
        advanced_metrics = advanced_analyzer.calculate_all_metrics()
        
        logger.info(f"Analysis completed for {player_name}")
    
    logger.info("Data analysis completed")

def create_visualizations(players=None):
    """
    Create player visualizations.
    
    Args:
        players (list, optional): List of player names.
                                  Defaults to potential breakout players.
    """
    logger.info("Starting visualization creation")
    
    # Determine which players to visualize
    if players is None:
        players_to_visualize = [p["name"] for p in POTENTIAL_BREAKOUT_PLAYERS]
        logger.info(f"Creating visualizations for all {len(players_to_visualize)} potential breakout players")
    else:
        players_to_visualize = players
        logger.info(f"Creating visualizations for {len(players_to_visualize)} specified players")
    
    # Create individual player cards
    for player_name in players_to_visualize:
        logger.info(f"Creating player card for {player_name}")
        
        # Create player card
        visualizer = PlayerCardVisualizer(player_name)
        try:
            visualizer.create_player_card()
            logger.info(f"Player card created for {player_name}")
        except Exception as e:
            logger.error(f"Error creating player card for {player_name}: {e}")
    
    # Create comparison visualizations
    logger.info("Creating comparison visualizations")
    comparison_visualizer = PlayerComparisonVisualizer(players_to_visualize)
    
    # Create breakout potential comparison
    try:
        comparison_visualizer.create_breakout_potential_comparison()
        logger.info("Breakout potential comparison created")
    except Exception as e:
        logger.error(f"Error creating breakout potential comparison: {e}")
    
    # Create impact comparison
    try:
        comparison_visualizer.create_impact_comparison()
        logger.info("Impact comparison created")
    except Exception as e:
        logger.error(f"Error creating impact comparison: {e}")
    
    # Create form and trajectory comparison
    try:
        comparison_visualizer.create_form_trajectory_comparison()
        logger.info("Form and trajectory comparison created")
    except Exception as e:
        logger.error(f"Error creating form and trajectory comparison: {e}")
    
    # Create top breakout candidates summary
    try:
        comparison_visualizer.create_top_breakout_candidates(top_n=5)
        logger.info("Top breakout candidates summary created")
    except Exception as e:
        logger.error(f"Error creating top breakout candidates summary: {e}")
    
    logger.info("Visualization creation completed")

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print(" DELHI CAPITALS BREAKOUT PLAYER ANALYSIS ".center(80, "="))
    print("="*80 + "\n")
    
    # Get players to analyze
    players_to_analyze = args.players
    
    # Run the complete pipeline if --all is specified
    if args.all:
        logger.info("Running complete analysis pipeline")
        collect_data(players_to_analyze)
        analyze_data(players_to_analyze)
        create_visualizations(players_to_analyze)
    else:
        # Run individual steps as specified
        if args.collect:
            collect_data(players_to_analyze)
        
        if args.analyze:
            analyze_data(players_to_analyze)
        
        if args.visualize:
            create_visualizations(players_to_analyze)
    
    # If no actions specified, print help
    if not (args.all or args.collect or args.analyze or args.visualize):
        print("No actions specified. Use --help for usage information.")
    
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()