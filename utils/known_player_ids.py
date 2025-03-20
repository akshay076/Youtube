"""
Known player IDs module for Delhi Capitals player analysis.
This module provides a lookup dictionary for player IDs on ESPNCricinfo.
"""

def get_known_player_id(player_name):
    """
    Return known ESPN Cricinfo player IDs for Delhi Capitals players.
    
    Args:
        player_name (str): Name of the player
        
    Returns:
        str: ESPN Cricinfo player ID or None if not found
    """
    known_ids = {
        # Key Players
        "Rishabh Pant": "931581",  # No longer with DC, but kept for reference
        "Axar Patel": "554691",    # Captain for IPL 2025
        "Kuldeep Yadav": "559235", # Retained player
        "Tristan Stubbs": "595978", # Retained player
        "Abishek Porel": "1223263", # Retained player
        
        # New Players for IPL 2025
        "KL Rahul": "422108",
        "Mitchell Starc": "311592",
        "Faf du Plessis": "44828",  # Vice-captain for IPL 2025
        
        # Other squad members
        "Prithvi Shaw": "1070168",
        "Mitchell Marsh": "272450",
        "Anrich Nortje": "481979",
        "Mukesh Kumar": "1070648",
        "Khaleel Ahmed": "926147",
        "Ishant Sharma": "236779",
        "Jhye Richardson": "774223",
        "Rasikh Dar": "1175401",
        "Sameer Rizvi": "1291970",
        "Sumit Kumar": "1203042",
        "Kumar Kushagra": "1223267",
        "Shai Hope": "581379",
        "Ashutosh Sharma": "1235617",
        "Ricky Bhui": "822313",
        "Harry Brook": "1020612",
        "Vicky Ostwal": "1239533",
        "Swastik Chikara": "1260285"
    }
    
    return known_ids.get(player_name)