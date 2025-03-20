"""
Comparison plots module for Delhi Capitals player analysis.
Creates visualizations comparing multiple players and their metrics.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_context("notebook", font_scale=1.2)

class PlayerComparisonVisualizer:
    """Class to create comparison visualizations for multiple players."""
    
    def __init__(self, player_names=None):
        """
        Initialize the player comparison visualizer.
        
        Args:
            player_names (list, optional): List of player names to compare.
                                           If None, uses potential breakout players.
        """
        if player_names is None:
            # Use all potential breakout players
            self.player_names = [p["name"] for p in POTENTIAL_BREAKOUT_PLAYERS]
        else:
            self.player_names = player_names
        
        # Load metrics for all players
        self.player_metrics = {}
        for player_name in self.player_names:
            # Load basic metrics
            basic_metrics_file = os.path.join(METRICS_DIR, f"{player_name.replace(' ', '_')}_basic_metrics.json")
            advanced_metrics_file = os.path.join(METRICS_DIR, f"{player_name.replace(' ', '_')}_advanced_metrics.json")
            
            try:
                with open(basic_metrics_file, 'r') as f:
                    basic_metrics = json.load(f)
                
                with open(advanced_metrics_file, 'r') as f:
                    advanced_metrics = json.load(f)
                
                self.player_metrics[player_name] = {
                    "basic": basic_metrics,
                    "advanced": advanced_metrics
                }
                logger.info(f"Loaded metrics for {player_name}")
            except FileNotFoundError:
                logger.warning(f"Metrics files for {player_name} not found")
        
        # Filter out players with missing metrics
        self.player_names = [name for name in self.player_names if name in self.player_metrics]
        logger.info(f"PlayerComparisonVisualizer initialized with {len(self.player_names)} players")
    
    def create_breakout_potential_comparison(self):
        """
        Create a comparison chart of breakout potential for all players.
        
        Returns:
            matplotlib.figure.Figure: The comparison figure
        """
        logger.info("Creating breakout potential comparison chart")
        
        if not self.player_names:
            logger.warning("No players with metrics data available")
            return None
        
        # Extract breakout potential scores and categories
        scores = []
        categories = []
        labels = []
        colors = []
        roles = []
        
        for player_name in self.player_names:
            metrics = self.player_metrics[player_name]
            breakout = metrics["advanced"].get("breakout_potential", {})
            
            score = breakout.get("breakout_potential_score", 0)
            category = breakout.get("breakout_category", "N/A")
            role = metrics["basic"].get("role", "Unknown")
            
            # Set color based on category
            if category == "Very High":
                color = '#2ecc71'  # Green
            elif category == "High":
                color = '#3498db'  # Blue
            elif category == "Moderate":
                color = '#f39c12'  # Orange
            elif category == "Low":
                color = '#e67e22'  # Dark orange
            else:
                color = '#e74c3c'  # Red
            
            # Store data
            scores.append(score)
            categories.append(category)
            labels.append(player_name)
            colors.append(color)
            roles.append(role)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set background color
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        # Sort players by breakout potential score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        sorted_categories = [categories[i] for i in sorted_indices]
        sorted_roles = [roles[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_labels, sorted_scores, color=sorted_colors, alpha=0.8)
        
        # Add category labels
        for i, (score, category) in enumerate(zip(sorted_scores, sorted_categories)):
            ax.text(score + 1, i, category, va='center', fontsize=10, fontweight='bold')
        
        # Add role labels
        for i, role in enumerate(sorted_roles):
            # Abbreviate role for display
            if "Batsman" in role and "WK" in role:
                role_abbr = "WK-Bat"
            elif "Batsman" in role:
                role_abbr = "Bat"
            elif "Bowler" in role:
                role_abbr = "Bowl"
            elif "All-rounder" in role:
                role_abbr = "All"
            else:
                role_abbr = role
            
            ax.text(-10, i, role_abbr, va='center', ha='right', fontsize=10)
        
        # Add labels and title
        ax.set_title('Delhi Capitals: Breakout Potential Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Breakout Potential Score (0-100)', fontsize=12)
        ax.set_xlim(-20, 120)  # Extra space for labels
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Remove y-axis but keep labels
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add breakout potential reference lines
        ax.axvline(x=80, color='#2ecc71', linestyle='--', alpha=0.5)
        ax.text(80, len(sorted_labels) + 0.5, 'Very High', fontsize=9, ha='center')
        
        ax.axvline(x=65, color='#3498db', linestyle='--', alpha=0.5)
        ax.text(65, len(sorted_labels) + 0.5, 'High', fontsize=9, ha='center')
        
        ax.axvline(x=50, color='#f39c12', linestyle='--', alpha=0.5)
        ax.text(50, len(sorted_labels) + 0.5, 'Moderate', fontsize=9, ha='center')
        
        ax.axvline(x=35, color='#e67e22', linestyle='--', alpha=0.5)
        ax.text(35, len(sorted_labels) + 0.5, 'Low', fontsize=9, ha='center')
        
        # Add team branding
        team_name = "DELHI CAPITALS"
        team_colors = ['#0078bc', '#f5455c']  # Delhi Capitals colors: blue and red
        plt.figtext(0.02, 0.02, team_name, fontsize=14, fontweight='bold', color=team_colors[0])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the comparison chart
        self.save_comparison_chart(fig, "breakout_potential_comparison")
        
        logger.info("Completed breakout potential comparison chart")
        return fig
    
    def create_top_breakout_candidates(self, top_n=5):
        """
        Create a summary visualization of the top breakout candidates.
        
        Args:
            top_n (int): Number of top candidates to include
            
        Returns:
            matplotlib.figure.Figure: The summary figure
        """
        logger.info(f"Creating top {top_n} breakout candidates visualization")
        
        if not self.player_names:
            logger.warning("No players with metrics data available")
            return None
        
        # Extract breakout potential and key metrics
        player_data = []
        for player_name in self.player_names:
            metrics = self.player_metrics[player_name]
            
            # Basic metrics
            role = metrics["basic"].get("role", "Unknown")
            batting = metrics["basic"].get("batting", {})
            matches = metrics["basic"].get("matches", 0)
            
            # Advanced metrics
            breakout = metrics["advanced"].get("breakout_potential", {})
            impact = metrics["advanced"].get("impact", {})
            form = metrics["advanced"].get("form", {})
            
            # Extract key metrics
            data = {
                "player_name": player_name,
                "role": role,
                "matches": matches,
                "breakout_score": breakout.get("breakout_potential_score", 0),
                "breakout_category": breakout.get("breakout_category", "N/A"),
                "impact_score": impact.get("overall_impact_score", 0),
                "batting_avg": batting.get("batting_average", 0),
                "strike_rate": batting.get("strike_rate", 0),
                "form_index": form.get("form_index", 0),
                "trajectory_score": form.get("trajectory_score", 0)
            }
            player_data.append(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(player_data)
        
        # Sort by breakout score and get top players
        df_top = df.sort_values("breakout_score", ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set background color
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        # Remove axis
        ax.axis('off')
        
        # Add title
        ax.set_title(f'Delhi Capitals: Top {top_n} Breakout Candidates', 
                   fontsize=16, fontweight='bold', y=0.98)
        
        # Create a table-like visualization
        cell_height = 0.15
        col_widths = [0.25, 0.1, 0.15, 0.15, 0.15, 0.2]
        col_positions = [sum(col_widths[:i]) for i in range(len(col_widths))]
        
        # Add headers
        headers = ["Player", "Role", "Breakout Score", "Impact", "Form", "Insight"]
        for i, header in enumerate(headers):
            ax.text(col_positions[i] + 0.02, 0.9, header, 
                   fontsize=12, fontweight='bold', ha='left', va='center')
        
        # Draw header separator line
        ax.plot([0, 1], [0.88, 0.88], 'k-', linewidth=1, alpha=0.3)
        
        # Add player rows
        for i, (_, player) in enumerate(df_top.iterrows()):
            y_pos = 0.85 - (i+1) * cell_height
            
            # Add player name
            ax.text(col_positions[0] + 0.02, y_pos, player["player_name"], 
                   fontsize=11, fontweight='bold', ha='left', va='center')
            
            # Add role
            role_text = player["role"]
            if len(role_text) > 12:
                role_text = role_text[:10] + "..."
            ax.text(col_positions[1] + 0.02, y_pos, role_text, 
                   fontsize=10, ha='left', va='center')
            
            # Add breakout score
            score = player["breakout_score"]
            category = player["breakout_category"]
            # Color based on category
            if category == "Very High":
                color = '#2ecc71'  # Green
            elif category == "High":
                color = '#3498db'  # Blue
            elif category == "Moderate":
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
                
            ax.text(col_positions[2] + 0.02, y_pos, f"{score:.1f}", 
                   fontsize=10, color=color, fontweight='bold', ha='left', va='center')
            ax.text(col_positions[2] + 0.07, y_pos, f"({category})", 
                   fontsize=9, color=color, ha='left', va='center')
            
            # Add impact score
            impact = player["impact_score"]
            ax.text(col_positions[3] + 0.02, y_pos, f"{impact:.1f}/100", 
                   fontsize=10, ha='left', va='center')
            
            # Add form index
            form = player["form_index"]
            trajectory = player["trajectory_score"]
            
            if trajectory >= 60:
                trajectory_text = "↗️"  # Improving
            elif trajectory >= 45:
                trajectory_text = "→"  # Stable
            else:
                trajectory_text = "↘️"  # Declining
                
            ax.text(col_positions[4] + 0.02, y_pos, f"{form:.1f} {trajectory_text}", 
                   fontsize=10, ha='left', va='center')
            
            # Add insight (custom for each player)
            # In a real implementation, this would be derived from the metrics
            if i == 0:
                insight = "Elite talent, consistent performer"
            elif i == 1:
                insight = "High upside, improving trajectory"
            elif i == 2:
                insight = "Good all-round skills, steady"
            elif i == 3:
                insight = "Raw talent, needs consistency"
            else:
                insight = "Potential, but development needed"
                
            ax.text(col_positions[5] + 0.02, y_pos, insight, 
                   fontsize=10, ha='left', va='center', style='italic')
            
            # Draw row separator line
            ax.plot([0, 1], [y_pos - 0.5*cell_height, y_pos - 0.5*cell_height], 
                   'k-', linewidth=1, alpha=0.1)
        
        # Add team branding
        team_name = "DELHI CAPITALS"
        team_colors = ['#0078bc', '#f5455c']  # Delhi Capitals colors: blue and red
        plt.figtext(0.02, 0.02, team_name, fontsize=14, fontweight='bold', color=team_colors[0])
        
        # Save the comparison chart
        self.save_comparison_chart(fig, "top_breakout_candidates")
        
        logger.info(f"Completed top {top_n} breakout candidates visualization")
        return fig
    
    def save_comparison_chart(self, fig, chart_name):
        """Save the comparison chart visualization to a file."""
        filename = os.path.join(VISUALIZATION_DIR, f"{chart_name}.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {chart_name} chart to {filename}")
    
    def create_impact_comparison(self):
        """
        Create a comparison chart of impact metrics for all players.
        
        Returns:
            matplotlib.figure.Figure: The comparison figure
        """
        logger.info("Creating impact metrics comparison chart")
        
        if not self.player_names:
            logger.warning("No players with metrics data available")
            return None
        
        # Extract impact metrics
        impact_data = []
        for player_name in self.player_names:
            metrics = self.player_metrics[player_name]
            impact = metrics["advanced"].get("impact", {})
            role = metrics["basic"].get("role", "Unknown")
            
            # Get impact metrics
            batting_impact = impact.get("batting_impact_index", 0)
            phase_impact = impact.get("phase_impact_score", 0)
            boundary_impact = impact.get("boundary_impact_score", 0)
            bowling_impact = impact.get("bowling_impact_index", 0)
            overall_impact = impact.get("overall_impact_score", 0)
            
            # Store data
            player_data = {
                "player_name": player_name,
                "role": role,
                "batting_impact": batting_impact,
                "phase_impact": phase_impact,
                "boundary_impact": boundary_impact,
                "bowling_impact": bowling_impact,
                "overall_impact": overall_impact
            }
            impact_data.append(player_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(impact_data)
        
        # Sort by overall impact
        df = df.sort_values("overall_impact", ascending=False)
        
        # Create figure for radar chart
        fig = plt.figure(figsize=(14, 10))
        
        # Set background color
        fig.patch.set_facecolor('#f8f9fa')
        
        # Create a grid for multiple radar charts (3x3 grid)
        max_players = 9
        num_players = min(len(df), max_players)
        
        # Calculate grid dimensions
        if num_players <= 3:
            grid_rows, grid_cols = 1, num_players
        elif num_players <= 6:
            grid_rows, grid_cols = 2, 3
        else:
            grid_rows, grid_cols = 3, 3
        
        # Add title
        fig.suptitle('Delhi Capitals: Player Impact Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        # Create radar charts for each player
        for i in range(num_players):
            player_data = df.iloc[i]
            player_name = player_data["player_name"]
            
            # Create subplot
            ax = fig.add_subplot(grid_rows, grid_cols, i+1, polar=True)
            
            # Define categories and values
            categories = ["Batting", "Phase Impact", "Boundary Impact", "Bowling", "Overall"]
            values = [
                player_data["batting_impact"],
                player_data["phase_impact"],
                player_data["boundary_impact"],
                player_data["bowling_impact"],
                player_data["overall_impact"]
            ]
            
            # Number of categories
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Values need to be normalized to 0-1 for the radar chart
            max_values = [100, 100, 100, 100, 100]  # Maximum expected values
            norm_values = [min(1.0, v / max_val) for v, max_val in zip(values, max_values)]
            norm_values += norm_values[:1]  # Close the loop
            
            # Draw the radar chart
            ax.plot(angles, norm_values, linewidth=2, linestyle='solid', color='#3498db')
            ax.fill(angles, norm_values, color='#3498db', alpha=0.4)
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            
            # Remove yticks
            ax.set_yticks([])
            
            # Add gridlines
            ax.grid(True, alpha=0.3)
            
            # Add player name as title
            role_abbr = ""
            if "Batsman" in player_data["role"] and "WK" in player_data["role"]:
                role_abbr = "(WK-Bat)"
            elif "Batsman" in player_data["role"]:
                role_abbr = "(Bat)"
            elif "Bowler" in player_data["role"]:
                role_abbr = "(Bowl)"
            elif "All-rounder" in player_data["role"]:
                role_abbr = "(All)"
            
            ax.set_title(f"{player_name} {role_abbr}", size=10, fontweight='bold')
            
            # Add value labels
            for j, value in enumerate(values):
                angle = angles[j]
                ax.text(angle, norm_values[j] + 0.05, f"{value:.1f}", 
                       fontsize=7, ha='center', va='center')
        
        # Add team branding
        team_name = "DELHI CAPITALS"
        team_colors = ['#0078bc', '#f5455c']  # Delhi Capitals colors: blue and red
        plt.figtext(0.02, 0.02, team_name, fontsize=14, fontweight='bold', color=team_colors[0])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the comparison chart
        self.save_comparison_chart(fig, "impact_comparison")
        
        logger.info("Completed impact metrics comparison chart")
        return fig
    
    def create_form_trajectory_comparison(self):
        """
        Create a comparison chart of form and trajectory metrics for all players.
        
        Returns:
            matplotlib.figure.Figure: The comparison figure
        """
        logger.info("Creating form and trajectory comparison chart")
        
        if not self.player_names:
            logger.warning("No players with metrics data available")
            return None
        
        # Extract form metrics
        form_data = []
        for player_name in self.player_names:
            metrics = self.player_metrics[player_name]
            form = metrics["advanced"].get("form", {})
            
            # Get form metrics
            form_index = form.get("form_index", 0)
            trajectory_score = form.get("trajectory_score", 0)
            consistency_score = form.get("consistency_score", 0)
            
            # Store data
            player_data = {
                "player_name": player_name,
                "form_index": form_index,
                "trajectory_score": trajectory_score,
                "consistency_score": consistency_score
            }
            form_data.append(player_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(form_data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Set background color
        fig.patch.set_facecolor('#f8f9fa')
        ax1.set_facecolor('#ffffff')
        ax2.set_facecolor('#ffffff')
        
        # 1. Create scatter plot of form vs trajectory
        scatter = ax1.scatter(df["form_index"], df["trajectory_score"], 
                            s=df["consistency_score"]*2, alpha=0.7, 
                            c=df["consistency_score"], cmap="viridis")
        
        # Add player name labels
        for i, player in enumerate(df["player_name"]):
            ax1.annotate(player, 
                       (df["form_index"].iloc[i], df["trajectory_score"].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add reference lines
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax1.text(150, 75, "Elite Prospects\n(High Form, Improving)", 
                fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.5))
        ax1.text(150, 25, "Recent Performers\n(High Form, Declining)", 
                fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.5))
        ax1.text(50, 75, "Emerging Talents\n(Lower Form, Improving)", 
                fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.5))
        ax1.text(50, 25, "Development Needed\n(Lower Form, Declining)", 
                fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.5))
        
        # Add labels and title
        ax1.set_title('Form vs. Trajectory Analysis', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Recent Form Index', fontsize=11)
        ax1.set_ylabel('Trajectory Score', fontsize=11)
        ax1.set_xlim(0, 200)
        ax1.set_ylim(0, 100)
        
        # Add colorbar for consistency
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Consistency Score', fontsize=10)
        
        # 2. Create bar chart of consistency scores
        # Sort by consistency
        df_sorted = df.sort_values("consistency_score", ascending=False)
        
        # Create horizontal bar chart
        bars = ax2.barh(df_sorted["player_name"], df_sorted["consistency_score"], 
                       color='#3498db', alpha=0.7)
        
        # Add labels and title
        ax2.set_title('Player Consistency Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Consistency Score (higher is better)', fontsize=11)
        ax2.set_xlim(0, 100)
        
        # Add grid lines
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Remove borders
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Add team branding
        team_name = "DELHI CAPITALS"
        team_colors = ['#0078bc', '#f5455c']  # Delhi Capitals colors: blue and red
        plt.figtext(0.02, 0.02, team_name, fontsize=14, fontweight='bold', color=team_colors[0])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the comparison chart
        self.save_comparison_chart(fig, "form_trajectory_comparison")
        
        logger.info("Completed form and trajectory comparison chart")
        return fig

# For testing - inside a conditional block to avoid execution when imported
if __name__ == "__main__":
    # Initialize comparison visualizer
    visualizer = PlayerComparisonVisualizer()
    
    # Create breakout potential comparison
    fig1 = visualizer.create_breakout_potential_comparison()
    
    # Create impact comparison
    fig2 = visualizer.create_impact_comparison()
    
    # Create form and trajectory comparison
    fig3 = visualizer.create_form_trajectory_comparison()
    
    # Create top breakout candidates summary
    fig4 = visualizer.create_top_breakout_candidates(top_n=5)
    
    # Show the figures
    plt.show()