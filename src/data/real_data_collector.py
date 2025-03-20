"""
Real data collection module for Delhi Capitals player analysis.
Fetches player data exclusively from ESPNCricinfo with no synthetic fallbacks.
"""

import os
import json
import pandas as pd
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

class RealDataCollector:
    """Class to collect cricket player data from ESPNCricinfo."""
    
    def __init__(self):
        """Initialize the data collector with necessary parameters."""
        self.headers = {'User-Agent': USER_AGENT}
        self.player_id_cache = {}
        self.load_cached_player_ids()
        logger.info("RealDataCollector initialized")
    
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
    
    def get_player_stats(self, player_info):
        """
        Get a player's statistics from ESPNCricinfo.
        
        Args:
            player_info (dict): Player information dictionary
            
        Returns:
            dict: Player statistics
        """
        player_name = player_info["name"]
        logger.info(f"Getting stats for player: {player_name}")
        
        # First, get the player ID
        player_id = self.search_player_id(player_name)
        
        if not player_id:
            logger.error(f"Could not find player ID for {player_name}. Cannot fetch data.")
            return None
        
        # Fetch player stats
        player_stats = self.fetch_player_stats(player_id, format_type="ipl")
        
        if player_stats:
            # Add player role from our data if not present in scraped data
            if "role" not in player_stats or not player_stats["role"]:
                player_stats["role"] = player_info["role"]
            
            # Add additional info if not present
            if "matches" not in player_stats or not player_stats["matches"]:
                player_stats["matches"] = player_info["experience"]
            
            logger.info(f"Successfully fetched stats for {player_name}")
            return player_stats
        else:
            logger.error(f"Failed to fetch stats for {player_name}")
            return None
    
    def fetch_player_stats(self, player_id, format_type="ipl"):
        """
        Fetch player statistics from ESPNCricinfo.
        
        Args:
            player_id (str): ESPNCricinfo player ID
            format_type (str): Type of cricket format (ipl, t20, odi, test)
            
        Returns:
            dict: Player statistics
        """
        logger.info(f"Fetching stats for player ID: {player_id}")
        
        try:
            # Construct API URLs
            profile_url = f"https://www.espncricinfo.com/player/{player_id}"
            stats_url = f"https://www.espncricinfo.com/ci/content/player/{player_id}.html"
            
            # Fetch profile page
            profile_response = requests.get(profile_url, headers=self.headers, timeout=10)
            profile_response.raise_for_status()
            
            # Parse profile data
            profile_soup = BeautifulSoup(profile_response.text, 'html.parser')
            
            # Extract player name
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
            
            # Extract player role
            role = profile_soup.select_one('.player-information .player-role')
            role_text = role.text.strip() if role else "Unknown"
            
            # Fetch statistics page
            stats_response = requests.get(stats_url, headers=self.headers, timeout=10)
            stats_response.raise_for_status()
            
            # Parse statistics data
            stats_soup = BeautifulSoup(stats_response.text, 'html.parser')
            
            # Extract statistics tables
            stats_tables = stats_soup.select('table.engineTable')
            
            # Initialize stats dictionary
            batting_stats = {}
            bowling_stats = {}
            
            # Process statistics tables
            for table in stats_tables:
                # Look for IPL or T20 table
                header = table.select_one('caption, th')
                if header and ('IPL' in header.text or 'Twenty20' in header.text):
                    # Find table header cells to determine column indices
                    headers = [th.text.strip() for th in table.select('thead th')]
                    
                    # Find the right row (usually the one with "matches" or "innings" in first column)
                    rows = table.select('tbody tr')
                    for row in rows:
                        cells = row.select('td')
                        if len(cells) >= 7:  # Typical batting stats table
                            # Extract data from cells based on headers
                            row_data = {}
                            for i, cell in enumerate(cells):
                                if i < len(headers):
                                    row_data[headers[i]] = cell.text.strip()
                            
                            # Determine if batting or bowling table
                            if 'Mat' in row_data and 'Inns' in row_data and 'Runs' in row_data:
                                # This is a batting table
                                batting_stats = row_data
                            elif 'Mat' in row_data and 'Wkts' in row_data and 'Econ' in row_data:
                                # This is a bowling table
                                bowling_stats = row_data
            
            # Extract match-by-match data if available (last 10 matches)
            match_data = []
            match_tables = stats_soup.select('table.match-by-match')
            
            for table in match_tables:
                if len(match_data) >= 10:  # Only get up to 10 most recent matches
                    break
                    
                # Find header row to determine column indices
                headers = [th.text.strip() for th in table.select('thead th')]
                
                # Process each match row
                match_rows = table.select('tbody tr')
                for row in match_rows:
                    cells = row.select('td')
                    if len(cells) >= 5:  # Minimum columns for match data
                        match_info = {}
                        for i, cell in enumerate(cells):
                            if i < len(headers):
                                match_info[headers[i]] = cell.text.strip()
                        
                        # Convert to our expected format
                        processed_match = self._process_match_data(match_info)
                        if processed_match:
                            match_data.append(processed_match)
                            if len(match_data) >= 10:
                                break
            
            # Calculate derived statistics
            innings = int(batting_stats.get('Inns', '0'))
            total_runs = int(batting_stats.get('Runs', '0'))
            not_outs = int(batting_stats.get('NO', '0'))
            
            # Batting average calculation
            try:
                batting_avg = float(batting_stats.get('Ave', '0'))
            except ValueError:
                batting_avg = total_runs / (innings - not_outs) if (innings - not_outs) > 0 else total_runs
            
            # Strike rate calculation
            try:
                strike_rate = float(batting_stats.get('SR', '0'))
            except ValueError:
                strike_rate = 0
            
            # Create player statistics dictionary
            player_stats = {
                "player_id": player_id,
                "player_name": player_name,
                "role": role_text,
                "matches": int(batting_stats.get('Mat', '0')),
                "innings": innings,
                "total_runs": total_runs,
                "average": batting_avg,
                "strike_rate": strike_rate,
                "not_outs": not_outs,
                "highest_score": batting_stats.get('HS', '0'),
                "total_fours": 0,  # Not typically available in summary stats
                "total_sixes": 0,  # Not typically available in summary stats
                "fifty_plus": int(batting_stats.get('50', '0')),
                "hundred_plus": int(batting_stats.get('100', '0')),
                "ducks": 0,  # Not typically available in summary stats
                
                # Bowling stats if available
                "total_wickets": int(bowling_stats.get('Wkts', '0')),
                "economy_rate": float(bowling_stats.get('Econ', '0')),
                "bowling_strike_rate": float(bowling_stats.get('SR', '0')),
                
                # Store match data
                "match_data": match_data
            }
            
            # Compute additional metrics that we need for analysis
            # Calculate total balls faced (if available from match data)
            total_balls_faced = sum(m.get("balls_faced", 0) for m in match_data if "balls_faced" in m)
            if total_balls_faced > 0:
                player_stats["total_balls_faced"] = total_balls_faced
            
            # Calculate boundary stats if available
            total_fours = sum(m.get("fours", 0) for m in match_data if "fours" in m)
            total_sixes = sum(m.get("sixes", 0) for m in match_data if "sixes" in m)
            
            if total_fours > 0 or total_sixes > 0:
                player_stats["total_fours"] = total_fours
                player_stats["total_sixes"] = total_sixes
                
                # Calculate boundary percentage if we have ball data
                if total_balls_faced > 0:
                    boundary_percentage = (total_fours + total_sixes) / total_balls_faced * 100
                    player_stats["boundary_percentage"] = round(boundary_percentage, 2)
            
            return player_stats
            
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return None
    
    def _process_match_data(self, match_info):
        """
        Process match info data into our expected format.
        
        Args:
            match_info (dict): Raw match info from cricinfo
            
        Returns:
            dict: Processed match data in our format
        """
        try:
            # Extract basic info
            date_str = match_info.get('Date', '')
            opposition = match_info.get('Against', '').replace('v ', '')
            runs_str = match_info.get('Runs', '0')
            
            # Handle special cases like "DNB" (Did Not Bat)
            if runs_str == 'DNB' or runs_str == '-':
                runs = 0
            else:
                runs = int(runs_str.replace('*', ''))  # Remove not out indicator
            
            # Extract dismissal info
            dismissal_type = "unknown"
            if '*' in runs_str:
                dismissal_type = "not out"
            elif runs == 0:
                dismissal_type = "duck"
            
            # Try to parse the date
            match_date = datetime.now()
            if date_str:
                try:
                    # Try different date formats
                    for fmt in ['%d %b %Y', '%b %d, %Y', '%Y-%m-%d']:
                        try:
                            match_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                except:
                    pass  # Use default date if parsing fails
            
            # Create match data entry
            match_data = {
                "match_id": f"match_{date_str}_{opposition}".replace(' ', '_'),
                "opposition": opposition,
                "date": match_date.strftime("%Y-%m-%d"),
                "runs": runs,
                "balls_faced": 0,  # Will be updated if available
                "strike_rate": 0,  # Will be calculated if balls faced is available
                "fours": 0,  # Will be updated if available
                "sixes": 0,  # Will be updated if available
                "dismissal_type": dismissal_type,
            }
            
            # Try to extract balls faced and calculate strike rate
            balls_faced_str = match_info.get('BF', '0')
            if balls_faced_str and balls_faced_str != '-':
                try:
                    balls_faced = int(balls_faced_str)
                    match_data["balls_faced"] = balls_faced
                    if balls_faced > 0 and runs > 0:
                        match_data["strike_rate"] = round((runs / balls_faced) * 100, 2)
                except ValueError:
                    pass
            
            # Try to extract boundaries if available
            fours_str = match_info.get('4s', '0')
            sixes_str = match_info.get('6s', '0')
            
            if fours_str and fours_str != '-':
                try:
                    match_data["fours"] = int(fours_str)
                except ValueError:
                    pass
                    
            if sixes_str and sixes_str != '-':
                try:
                    match_data["sixes"] = int(sixes_str)
                except ValueError:
                    pass
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error processing match data: {e}")
            return None
    
    def collect_all_player_data(self, players_list=None):
        """
        Collect data for all players in the list.
        
        Args:
            players_list (list, optional): List of player info dictionaries.
                                          Defaults to DELHI_CAPITALS_SQUAD.
            
        Returns:
            dict: Dictionary of player data
        """
        if players_list is None:
            players_list = DELHI_CAPITALS_SQUAD
            
        logger.info(f"Collecting data for {len(players_list)} players")
        all_player_data = {}
        
        for player_info in players_list:
            player_name = player_info["name"]
            
            # Get player data
            player_data = self.get_player_stats(player_info)
            
            if player_data:
                all_player_data[player_name] = player_data
                
                # Save individual player data
                self.save_player_data(player_name, player_data)
            else:
                logger.warning(f"Could not collect data for player: {player_name}")
        
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