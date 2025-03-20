"""
Improved data collection module for Delhi Capitals player analysis.
Uses multiple methods to reliably find player IDs and statistics.
"""

import os
import json
import pandas as pd
import numpy as np
import requests
import re
import csv
import sys
import logging
from bs4 import BeautifulSoup
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedDataCollector:
    """Class to collect cricket player data from ESPNCricinfo."""
    
    def __init__(self):
        """Initialize the data collector with necessary parameters."""
        self.headers = {'User-Agent': USER_AGENT}
        self.player_id_cache = {}
        self.load_cached_player_ids()
        logger.info("ImprovedDataCollector initialized")
    
    def load_cached_player_ids(self):
        """Load previously cached player IDs from CSV file."""
        cache_file = os.path.join(DATA_DIR, "player_ids_cache.csv")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.player_id_cache[row['name']] = row['player_id']
                logger.info(f"Loaded {len(self.player_id_cache)} player IDs from cache")
            except Exception as e:
                logger.error(f"Error loading player ID cache: {e}")
    
    def save_player_id_to_cache(self, player_name, player_id):
        """Save a player ID to the cache."""
        self.player_id_cache[player_name] = player_id
        
        # Save to CSV
        cache_file = os.path.join(DATA_DIR, "player_ids_cache.csv")
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(cache_file)
        
        with open(cache_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'player_id'])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'name': player_name,
                'player_id': player_id
            })
        
        logger.info(f"Saved player ID for {player_name} to cache")
    
    def search_player_id(self, player_name):
        """
        Search for a player's ID on ESPNCricinfo using multiple methods.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            str: Player ID if found, None otherwise
        """
        logger.info(f"Searching for player ID: {player_name}")
        
        # Method 1: Check cache first
        if player_name in self.player_id_cache:
            player_id = self.player_id_cache[player_name]
            logger.info(f"Found player ID for {player_name} in cache: {player_id}")
            return player_id
        
        # Method 2: Check known player IDs
        try:
            from utils.known_player_ids import get_known_player_id
            player_id = get_known_player_id(player_name)
            
            if player_id:
                logger.info(f"Found player ID for {player_name} in known IDs: {player_id}")
                self.save_player_id_to_cache(player_name, player_id)
                return player_id
        except ImportError:
            logger.warning("known_player_ids module not found, skipping this method")
        
        # Method 3: Direct URL construction
        player_id = self.get_player_id_from_direct_url(player_name)
        if player_id:
            self.save_player_id_to_cache(player_name, player_id)
            return player_id
        
        # Method 4: Search ESPNCricinfo
        player_id = self.get_player_id_from_search(player_name)
        if player_id:
            self.save_player_id_to_cache(player_name, player_id)
            return player_id
        
        logger.warning(f"Could not find ID for player: {player_name}")
        return None
    
    def get_player_id_from_direct_url(self, player_name):
        """
        Get player ID by trying direct URL patterns.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            str: Player ID if found, None otherwise
        """
        # Format name for URL (lowercase, spaces to hyphens)
        url_name = player_name.lower().replace(" ", "-")
        
        # Define URLs to try
        urls_to_try = [
            f"https://www.espncricinfo.com/player/{url_name}",
            f"https://www.espncricinfo.com/cricketers/{url_name}"
        ]
        
        for url in urls_to_try:
            try:
                logger.info(f"Trying direct URL: {url}")
                response = requests.get(url, headers=self.headers, timeout=10)
                
                # If redirected to a player page, extract the ID from the final URL
                if response.status_code == 200:
                    # Extract the ID from the URL or page content
                    match = re.search(r'/player/(\d+)', response.url)
                    if match:
                        player_id = match.group(1)
                        logger.info(f"Found player ID for {player_name} via direct URL: {player_id}")
                        return player_id
                    
                    # If not in URL, try to find in page content
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = soup.select('a[href*="/player/"]')
                    
                    for link in links:
                        href = link.get('href')
                        match = re.search(r'/player/(\d+)', href)
                        if match:
                            player_id = match.group(1)
                            logger.info(f"Found player ID for {player_name} in page content: {player_id}")
                            return player_id
            
            except Exception as e:
                logger.error(f"Error retrieving {url}: {e}")
                
            # Be nice to the server
            time.sleep(1)
        
        return None
    
    def get_player_id_from_search(self, player_name):
        """
        Get player ID by searching ESPNCricinfo.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            str: Player ID if found, None otherwise
        """
        try:
            # Format search query
            search_query = player_name.replace(" ", "+")
            search_url = f"https://www.espncricinfo.com/search?query={search_query}"
            
            logger.info(f"Searching Cricinfo: {search_url}")
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find player links in search results
                player_links = soup.select('a[href*="/player/"]')
                
                for link in player_links:
                    href = link.get('href')
                    match = re.search(r'/player/(\d+)', href)
                    if match:
                        player_id = match.group(1)
                        
                        # Verify this is the right player by checking the name on the player page
                        verification = self.verify_player_id(player_id, player_name)
                        if verification:
                            logger.info(f"Found and verified player ID for {player_name}: {player_id}")
                            return player_id
        
        except Exception as e:
            logger.error(f"Error searching for player: {e}")
        
        return None
    
    def verify_player_id(self, player_id, player_name):
        """
        Verify a player ID matches the player name.
        
        Args:
            player_id (str): Player ID to verify
            player_name (str): Expected player name
            
        Returns:
            bool: True if verified, False otherwise
        """
        try:
            player_url = f"https://www.espncricinfo.com/player/{player_id}"
            response = requests.get(player_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for player name in title or heading
                title = soup.find('title')
                if title and player_name.lower() in title.text.lower():
                    return True
                
                # Check page headings
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
                for heading in headings:
                    if player_name.lower() in heading.text.lower():
                        return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error verifying player ID: {e}")
            return False
    
    def get_player_stats(self, player_id, format_type="ipl"):
        """
        Get a player's statistics from ESPNCricinfo.
        In a real implementation, this would scrape actual stats.
        For this demo, we'll still use synthetic data.
        
        Args:
            player_id (str): ESPNCricinfo player ID
            format_type (str): Type of cricket format (ipl, t20, odi, test)
            
        Returns:
            dict: Player statistics
        """
        logger.info(f"Getting stats for player ID: {player_id}")
        
        # Find player info by ID
        player_name = None
        for player in DELHI_CAPITALS_SQUAD:
            if self.search_player_id(player["name"]) == player_id:
                player_name = player["name"]
                player_info = player
                break
        
        if not player_name:
            logger.warning(f"Could not find player info for ID: {player_id}")
            return {}
        
        # In a real implementation, we would scrape the actual player stats
        # For this demo, we'll generate synthetic data as before
        
        # Generate synthetic match data with consistent player ID
        match_data = self.generate_synthetic_match_data(player_info)
        
        # Create player statistics dictionary
        innings_played = len([m for m in match_data if m["runs"] > 0 or m["balls_faced"] > 0])
        total_runs = sum(m["runs"] for m in match_data)
        total_balls = sum(m["balls_faced"] for m in match_data)
        total_fours = sum(m["fours"] for m in match_data)
        total_sixes = sum(m["sixes"] for m in match_data)
        not_outs = sum(1 for m in match_data if m["dismissal_type"] == "not out")
        
        # Batting average calculation
        average = total_runs / (innings_played - not_outs) if (innings_played - not_outs) > 0 else total_runs
        
        # Strike rate calculation
        strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
        
        # Calculate player type based on role
        if "Batsman" in player_info["role"]:
            player_type = "Batsman"
        elif "Bowler" in player_info["role"]:
            player_type = "Bowler"
        elif "All-rounder" in player_info["role"]:
            player_type = "All-rounder"
        elif "WK" in player_info["role"]:
            player_type = "Wicketkeeper-Batsman"
        else:
            player_type = "Unknown"
        
        # Summary dictionary
        player_stats = {
            "player_id": player_id,
            "player_name": player_name,
            "role": player_info["role"],
            "player_type": player_type,
            "matches": player_info["experience"],
            "innings": innings_played,
            "total_runs": total_runs,
            "total_balls_faced": total_balls,
            "average": round(average, 2),
            "strike_rate": round(strike_rate, 2),
            "total_fours": total_fours,
            "total_sixes": total_sixes,
            "not_outs": not_outs,
            "boundary_percentage": round((total_fours + total_sixes) / total_balls * 100, 2) if total_balls > 0 else 0,
            "highest_score": max([m["runs"] for m in match_data]) if match_data else 0,
            "ducks": sum(1 for m in match_data if m["runs"] == 0 and m["dismissal_type"] != "not out"),
            "fifty_plus": sum(1 for m in match_data if m["runs"] >= 50),
            "hundred_plus": sum(1 for m in match_data if m["runs"] >= 100),
            
            # Bowling stats (if applicable)
            "total_wickets": sum(m["wickets"] for m in match_data),
            "economy_rate": round(np.mean([m["economy"] for m in match_data if "economy" in m]), 2),
            "bowling_strike_rate": round(np.mean([m["bowling_strike_rate"] for m in match_data if "bowling_strike_rate" in m]), 2),
            
            # Store full match data
            "match_data": match_data
        }
        
        return player_stats
    
    def generate_synthetic_match_data(self, player_info):
        """
        Generate synthetic match data for a player based on their role and experience.
        This is the same method as in the original DataCollector.
        
        Args:
            player_info (dict): Player information including role and experience
            
        Returns:
            list: List of match data dictionaries
        """
        # Same implementation as before (reuse existing code)
        # For brevity, this implementation is omitted here
        # In a real implementation, you would integrate with actual data sources
        
        # Placeholder for demo
        return []  # Replace with actual implementation
    
    def collect_all_player_data(self, players_list=None):
        """
        Collect data for all players in the list.
        
        Args:
            players_list (list, optional): List of player info dictionaries.
                                          Defaults to DELHI_CAPITALS_SQUAD.
            
        Returns:
            dict: Dictionary of player data
        """
        # Same implementation as before with improved player ID lookup
        
        if players_list is None:
            players_list = DELHI_CAPITALS_SQUAD
            
        logger.info(f"Collecting data for {len(players_list)} players")
        all_player_data = {}
        
        for player_info in players_list:
            player_name = player_info["name"]
            player_id = self.search_player_id(player_name)
            
            if player_id:
                player_data = self.get_player_stats(player_id)
                all_player_data[player_name] = player_data
                
                # Save individual player data
                self.save_player_data(player_name, player_data)
            else:
                logger.warning(f"Could not find ID for player: {player_name}, skipping data collection")
        
        # Save combined data
        self.save_combined_player_data(all_player_data)
        
        logger.info(f"Completed data collection for {len(all_player_data)} players")
        return all_player_data
    
    def save_player_data(self, player_name, player_data):
        """
        Save player data to a JSON file.
        
        Args:
            player_name (str): Name of the player
            player_data (dict): Player statistics
        """
        filename = os.path.join(RAW_DATA_DIR, f"{player_name.replace(' ', '_')}.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(player_data, f, indent=4)
        
        logger.info(f"Saved data for {player_name} to {filename}")
    
    def save_combined_player_data(self, all_player_data):
        """
        Save combined player data to a JSON file.
        
        Args:
            all_player_data (dict): Dictionary of all player data
        """
        filename = os.path.join(RAW_DATA_DIR, "delhi_capitals_all_players.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(all_player_data, f, indent=4)
        
        logger.info(f"Saved combined player data to {filename}")
    
    def load_player_data(self, player_name):
        """
        Load player data from a JSON file.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            dict: Player statistics
        """
        filename = os.path.join(RAW_DATA_DIR, f"{player_name.replace(' ', '_')}.json")
        
        try:
            with open(filename, 'r') as f:
                player_data = json.load(f)
            logger.info(f"Loaded data for {player_name} from {filename}")
            return player_data
        except FileNotFoundError:
            logger.warning(f"Data file for {player_name} not found")
            return None
    
    def load_all_player_data(self):
        """
        Load all player data from the combined JSON file.
        
        Returns:
            dict: Dictionary of all player data
        """
        filename = os.path.join(RAW_DATA_DIR, "delhi_capitals_all_players.json")
        
        try:
            with open(filename, 'r') as f:
                all_player_data = json.load(f)
            logger.info(f"Loaded combined player data from {filename}")
            return all_player_data
        except FileNotFoundError:
            logger.warning("Combined player data file not found")
            return None
    
    def fetch_real_player_stats(self, player_id, format_type="ipl"):
        """
        Fetch actual player statistics from ESPNCricinfo API or website.
        This is a template for implementing real data collection.
        
        Args:
            player_id (str): ESPNCricinfo player ID
            format_type (str): Type of cricket format (ipl, t20, odi, test)
            
        Returns:
            dict: Player statistics
        """
        logger.info(f"Fetching real stats for player ID: {player_id}")
        
        try:
            # Construct API URLs
            profile_url = f"https://www.espncricinfo.com/player/{player_id}"
            stats_url = f"https://www.espncricinfo.com/ci/content/player/{player_id}.html"
            
            # Fetch profile page
            profile_response = requests.get(profile_url, headers=self.headers, timeout=10)
            profile_response.raise_for_status()
            
            # Parse profile data
            profile_soup = BeautifulSoup(profile_response.text, 'html.parser')
            
            # Example: extract player name
            player_name = profile_soup.select_one('h1.player-name')
            if player_name:
                player_name = player_name.text.strip()
            else:
                # Fallback method to find player name
                title = profile_soup.find('title')
                if title:
                    player_name = title.text.split(' | ')[0].strip()
                else:
                    player_name = "Unknown"
            
            # Example: extract player role
            role = profile_soup.select_one('.player-information .player-role')
            role_text = role.text.strip() if role else "Unknown"
            
            # Fetch statistics page
            stats_response = requests.get(stats_url, headers=self.headers, timeout=10)
            stats_response.raise_for_status()
            
            # Parse statistics data
            stats_soup = BeautifulSoup(stats_response.text, 'html.parser')
            
            # Extract statistics tables
            stats_tables = stats_soup.select('table.engineTable')
            
            # Process statistics - this would be tailored to the actual HTML structure
            # For IPL statistics, find the appropriate table
            ipl_stats = {}
            for table in stats_tables:
                # Look for IPL or T20 table
                header = table.select_one('caption, th')
                if header and ('IPL' in header.text or 'T20' in header.text):
                    # Process table rows
                    rows = table.select('tbody tr')
                    for row in rows:
                        cells = row.select('td')
                        if len(cells) >= 7:  # Typical batting stats table
                            # Extract data from cells
                            # This would be customized based on the actual table structure
                            pass
            
            # Construct and return player data
            # This would be completed with actual data parsing
            return {
                "player_id": player_id,
                "player_name": player_name,
                "role": role_text,
                # Additional stats would be filled in
            }
            
        except Exception as e:
            logger.error(f"Error fetching real player stats: {e}")
            return {}
    
    def generate_synthetic_match_data(self, player_info):
        """
        Generate synthetic match data for a player based on their role and experience.
        This replicates the functionality from the original DataCollector.
        
        Args:
            player_info (dict): Player information including role and experience
            
        Returns:
            list: List of match data dictionaries
        """
        logger.info(f"Generating synthetic match data for {player_info['name']}")
        
        player_name = player_info["name"]
        matches = player_info["experience"]
        match_data = []
        
        # Set base parameters based on player role
        if "Batsman" in player_info["role"]:
            avg_runs = 25
            avg_sr = 140
            
            # Special cases for known batsmen
            if player_name == "Jake Fraser-McGurk":
                avg_runs = 35
                avg_sr = 200
            elif player_name == "Rishabh Pant":
                avg_runs = 30
                avg_sr = 155
            elif player_name == "Harry Brook":
                avg_runs = 28
                avg_sr = 145
                
        elif "All-rounder" in player_info["role"]:
            avg_runs = 20
            avg_sr = 135
            
            # Special cases for known all-rounders
            if player_name == "Axar Patel":
                avg_runs = 18
                avg_sr = 140
            elif player_name == "Mitchell Marsh":
                avg_runs = 25
                avg_sr = 150
                
        else:  # Bowler
            avg_runs = 8
            avg_sr = 120
        
        # Opponent teams (excluding Delhi Capitals)
        teams = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", 
                "Kolkata Knight Riders", "Sunrisers Hyderabad", "Punjab Kings", 
                "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants"]
        
        # Generate match data with realistic distribution
        for i in range(matches):
            # More recent matches have higher probability of better performance
            recency_boost = 1 + (i / max(1, matches)) * 0.5
            
            # Opposition team
            opposition = random.choice(teams)
            
            # Adjust opposition strength based on team (1-10 scale)
            opposition_strength_map = {
                "Mumbai Indians": 8,
                "Chennai Super Kings": 9,
                "Royal Challengers Bangalore": 7,
                "Kolkata Knight Riders": 8,
                "Sunrisers Hyderabad": 7,
                "Punjab Kings": 6,
                "Rajasthan Royals": 7,
                "Gujarat Titans": 8,
                "Lucknow Super Giants": 7
            }
            opposition_strength = opposition_strength_map.get(opposition, 7)
            
            # Add some randomness
            opposition_strength = max(1, min(10, opposition_strength + random.randint(-1, 1)))
            
            # Some randomness to simulate match-to-match variation
            runs = max(0, int(np.random.normal(avg_runs * recency_boost, avg_runs * 0.4)))
            
            # Strike rate varies based on runs
            sr_boost = 1.2 if runs > avg_runs else 0.8
            strike_rate = max(100, np.random.normal(avg_sr * sr_boost, 20))
            
            # Boundaries calculation
            fours = int(runs * 0.4 / 4) if runs > 0 else 0
            sixes = int(runs * 0.3 / 6) if runs > 0 else 0
            
            # Ensure runs match boundaries
            boundary_runs = (fours * 4) + (sixes * 6)
            if boundary_runs > runs:
                # Adjust if boundaries exceed total runs
                excess = boundary_runs - runs
                while excess >= 4 and fours > 0:
                    fours -= 1
                    excess -= 4
                while excess >= 6 and sixes > 0:
                    sixes -= 1
                    excess -= 6
            
            # Match context and phases
            powerplay_runs = int(runs * 0.4) if "Batsman" in player_info["role"] else int(runs * 0.3)
            middle_runs = int(runs * 0.4)
            death_runs = runs - powerplay_runs - middle_runs
            
            # Venue and match pressure
            home_game = random.choice([True, False])
            high_pressure = random.choice([True, False], p=[0.3, 0.7])
            
            # Ball-by-ball data would be more detailed in a real implementation
            balls_faced = max(1, int(runs / (strike_rate/100))) if runs > 0 else random.randint(1, 10)
            
            # Wickets data for bowlers
            wickets = 0
            economy = 0
            bowling_strike_rate = 0
            
            if "Bowler" in player_info["role"] or "All-rounder" in player_info["role"]:
                # Generate bowling data
                wickets = np.random.binomial(4, 0.25)  # Average around 1 wicket per match
                overs = random.uniform(2, 4)
                runs_conceded = int(overs * random.uniform(7, 11))  # Economy between 7-11
                
                economy = runs_conceded / overs if overs > 0 else 0
                bowling_strike_rate = (overs * 6) / wickets if wickets > 0 else (overs * 6)
            
            # Generate a random date in IPL 2024 season (March-May 2024)
            match_date = datetime(2024, 3, 22) + timedelta(days=i*3 % 60)
            
            match_data.append({
                "match_id": f"match_{i+1}",
                "opposition": opposition,
                "date": match_date.strftime("%Y-%m-%d"),
                "runs": runs,
                "balls_faced": balls_faced,
                "strike_rate": round(strike_rate, 2),
                "fours": fours,
                "sixes": sixes,
                "dismissal_type": random.choice(["caught", "bowled", "lbw", "run out", "not out"]),
                "opposition_strength": opposition_strength,
                "powerplay_runs": powerplay_runs,
                "middle_runs": middle_runs,
                "death_runs": death_runs,
                "home_game": home_game,
                "high_pressure": high_pressure,
                "wickets": wickets,
                "economy": round(economy, 2),
                "bowling_strike_rate": round(bowling_strike_rate, 2)
            })
        
        # Sort by date (most recent first)
        match_data.sort(key=lambda x: x["date"], reverse=True)
        logger.info(f"Generated {len(match_data)} matches of synthetic data for {player_name}")
        
        return match_data

# For testing
if __name__ == "__main__":
    # Test with a specific player
    collector = ImprovedDataCollector()
    
    # Test player ID lookup
    test_player = "Rishabh Pant"
    player_id = collector.search_player_id(test_player)
    print(f"Player ID for {test_player}: {player_id}")
    
    # Test data collection for a single player
    if player_id:
        player_data = collector.get_player_stats(player_id)
        print(f"Generated data for {test_player}:")
        print(json.dumps(player_data, indent=2)[:500] + "...")  # Print first 500 chars
    
    # Test collecting data for all potential breakout players
    collector.collect_all_player_data(POTENTIAL_BREAKOUT_PLAYERS[:3])  # Just test with first 3 players