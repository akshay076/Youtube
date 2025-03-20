"""
Player card visualization module for Delhi Capitals player analysis.
Creates visual player cards with performance metrics and breakout potential.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import sys
import logging
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_context("notebook", font_scale=1.2)

class PlayerCardVisualizer:
    """Class to create visual player cards with performance metrics."""
    
    def __init__(self, player_name, basic_metrics=None, advanced_metrics=None):
        """
        Initialize the player card visualizer.
        
        Args:
            player_name (str): Name of the player
            basic_metrics (dict, optional): Basic metrics data
            advanced_metrics (dict, optional): Advanced metrics data
        """
        self.player_name = player_name
        
        # Load basic metrics if not provided
        if basic_metrics:
            self.basic_metrics = basic_metrics
        else:
            metrics_file = os.path.join(METRICS_DIR, f"{player_name.replace(' ', '_')}_basic_metrics.json")
            try:
                with open(metrics_file, 'r') as f:
                    self.basic_metrics = json.load(f)
                logger.info(f"Loaded basic metrics for {player_name} from file")
            except FileNotFoundError:
                logger.warning(f"Basic metrics file for {player_name} not found")
                self.basic_metrics = {}
        
        # Load advanced metrics if not provided
        if advanced_metrics:
            self.advanced_metrics = advanced_metrics
        else:
            metrics_file = os.path.join(METRICS_DIR, f"{player_name.replace(' ', '_')}_advanced_metrics.json")
            try:
                with open(metrics_file, 'r') as f:
                    self.advanced_metrics = json.load(f)
                logger.info(f"Loaded advanced metrics for {player_name} from file")
            except FileNotFoundError:
                logger.warning(f"Advanced metrics file for {player_name} not found")
                self.advanced_metrics = {}
        
        logger.info(f"PlayerCardVisualizer initialized for {player_name}")
    
    def create_player_card(self):
        """
        Create a comprehensive player card visualization.
        
        Returns:
            matplotlib.figure.Figure: The player card figure
        """
        logger.info(f"Creating player card for {self.player_name}")
        
        # Get player role
        role = self.basic_metrics.get("role", "Unknown")
        
        # Create figure with appropriate size
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(4, 3, figure=fig)
        
        # Set background color
        fig.patch.set_facecolor('#f8f9fa')
        
        # Add title with player info
        matches = self.basic_metrics.get("matches", 0)
        fig.suptitle(f"{self.player_name} - {role} (IPL Experience: {matches} matches)", 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Add Delhi Capitals branding
        team_colors = ['#0078bc', '#f5455c']  # Delhi Capitals colors: blue and red
        
        # Create header panel
        header_ax = fig.add_subplot(gs[0, :])
        header_ax.set_facecolor(team_colors[0])
        header_ax.set_xlim(0, 1)
        header_ax.set_ylim(0, 1)
        header_ax.axis('off')
        
        # Add player role icon based on role
        icon_pos = 0.05
        if "Batsman" in role:
            header_ax.text(icon_pos, 0.5, "🏏", fontsize=24, va='center')
        elif "Bowler" in role:
            header_ax.text(icon_pos, 0.5, "🎯", fontsize=24, va='center')
        elif "All-rounder" in role:
            header_ax.text(icon_pos, 0.5, "🏏🎯", fontsize=24, va='center')
        elif "WK" in role:
            header_ax.text(icon_pos, 0.5, "🧤", fontsize=24, va='center')
        
        # Add player type text
        header_ax.text(0.5, 0.5, f"Delhi Capitals - Potential Breakout Player Analysis", 
                      fontsize=16, color='white', ha='center', va='center')
        
        # Add breakout potential score if available
        breakout_potential = self.advanced_metrics.get("breakout_potential", {})
        score = breakout_potential.get("breakout_potential_score", 0)
        category = breakout_potential.get("breakout_category", "N/A")
        
        if score > 0:
            # Create a colored circle based on the score
            if score >= 80:
                color = '#2ecc71'  # Green for very high
            elif score >= 65:
                color = '#3498db'  # Blue for high
            elif score >= 50:
                color = '#f39c12'  # Orange for moderate
            elif score >= 35:
                color = '#e67e22'  # Dark orange for low
            else:
                color = '#e74c3c'  # Red for very low
            
            # Add score circle
            circle = patches.Circle((0.9, 0.5), 0.08, facecolor=color, edgecolor='white', linewidth=2)
            header_ax.add_patch(circle)
            header_ax.text(0.9, 0.5, f"{int(score)}", fontsize=14, color='white', 
                          ha='center', va='center', fontweight='bold')
            
            # Add category text below
            header_ax.text(0.9, 0.3, f"{category}", fontsize=10, color='white', 
                          ha='center', va='center')
            header_ax.text(0.9, 0.2, "Potential", fontsize=8, color='white', 
                          ha='center', va='center')
        
        # 1. Key Performance Metrics Panel
        perf_ax = fig.add_subplot(gs[1, 0])
        self._add_key_metrics_panel(perf_ax)
        
        # 2. Form and Trajectory Panel
        form_ax = fig.add_subplot(gs[1, 1])
        self._add_form_trajectory_panel(form_ax)
        
        # 3. Phase Analysis Panel
        phase_ax = fig.add_subplot(gs[1, 2])
        self._add_phase_analysis_panel(phase_ax)
        
        # 4. Impact Metrics Panel
        impact_ax = fig.add_subplot(gs[2, 0])
        self._add_impact_metrics_panel(impact_ax)
        
        # 5. Opposition Quality Panel
        opposition_ax = fig.add_subplot(gs[2, 1])
        self._add_opposition_quality_panel(opposition_ax)
        
        # 6. Venue Analysis Panel
        venue_ax = fig.add_subplot(gs[2, 2])
        self._add_venue_analysis_panel(venue_ax)
        
        # 7. Breakout Potential Panel - spans bottom row
        breakout_ax = fig.add_subplot(gs[3, :])
        self._add_breakout_potential_panel(breakout_ax)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the player card
        self.save_player_card(fig)
        
        logger.info(f"Completed player card creation for {self.player_name}")
        return fig
    
    def _add_key_metrics_panel(self, ax):
        """Add key performance metrics panel to the player card."""
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Key Performance Metrics", fontsize=12, fontweight='bold', ha='center')
        
        # Get batting metrics
        batting = self.basic_metrics.get("batting", {})
        matches = self.basic_metrics.get("matches", 0)
        avg = batting.get("batting_average", 0)
        sr = batting.get("strike_rate", 0)
        
        # Get bowling metrics if applicable
        bowling = self.basic_metrics.get("bowling", {})
        economy = bowling.get("economy_rate", 0)
        wickets = bowling.get("total_wickets", 0)
        
        # Player role determines which metrics to show
        role = self.basic_metrics.get("role", "")
        
        y_pos = 0.8
        
        # Add batting metrics for everyone
        ax.text(0.05, y_pos, "Batting:", fontsize=10, fontweight='bold')
        y_pos -= 0.1
        ax.text(0.1, y_pos, f"Matches: {matches}", fontsize=9)
        y_pos -= 0.08
        ax.text(0.1, y_pos, f"Average: {avg}", fontsize=9)
        y_pos -= 0.08
        ax.text(0.1, y_pos, f"Strike Rate: {sr}", fontsize=9)
        y_pos -= 0.08
        
        # Add boundary stats
        boundary_pct = batting.get("boundary_percentage", 0)
        fours = batting.get("total_fours", 0)
        sixes = batting.get("total_sixes", 0)
        ax.text(0.1, y_pos, f"Boundary %: {boundary_pct}%", fontsize=9)
        y_pos -= 0.08
        ax.text(0.1, y_pos, f"4s/6s: {fours}/{sixes}", fontsize=9)
        y_pos -= 0.1
        
        # Add bowling metrics if applicable
        if "Bowler" in role or "All-rounder" in role:
            ax.text(0.05, y_pos, "Bowling:", fontsize=10, fontweight='bold')
            y_pos -= 0.1
            ax.text(0.1, y_pos, f"Wickets: {wickets}", fontsize=9)
            y_pos -= 0.08
            ax.text(0.1, y_pos, f"Economy: {economy}", fontsize=9)
            y_pos -= 0.08
            
            # Add wicket taking ability
            if matches > 0:
                wickets_per_match = wickets / matches
                ax.text(0.1, y_pos, f"Wickets/Match: {wickets_per_match:.2f}", fontsize=9)
    
    def _add_form_trajectory_panel(self, ax):
        """Add form and trajectory analysis panel to the player card."""
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Form & Trajectory Analysis", fontsize=12, fontweight='bold', ha='center')
        
        # Get form metrics
        form = self.advanced_metrics.get("form", {})
        
        # Draw metrics as colored bars
        metrics = [
            ("Recent Form", form.get("form_index", 0) / 100),
            ("Career Trajectory", form.get("trajectory_score", 0) / 100),
            ("Consistency", form.get("consistency_score", 0) / 100)
        ]
        
        y_pos = 0.75
        bar_height = 0.08
        
        for label, value in metrics:
            # Add label
            ax.text(0.05, y_pos, label, fontsize=9)
            
            # Draw background bar
            ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9, bar_height, 
                                         facecolor='#e0e0e0', edgecolor='none', alpha=0.5))
            
            # Draw value bar with color based on value
            if value >= 0.8:
                color = '#2ecc71'  # Green
            elif value >= 0.6:
                color = '#3498db'  # Blue
            elif value >= 0.4:
                color = '#f39c12'  # Orange
            elif value >= 0.2:
                color = '#e67e22'  # Dark orange
            else:
                color = '#e74c3c'  # Red
            
            ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9*value, bar_height, 
                                         facecolor=color, edgecolor='none', alpha=0.8))
            
            # Add value text
            ax.text(0.05 + 0.9*value + 0.02, y_pos - bar_height/2, f"{int(value*100)}", 
                   fontsize=9, va='center')
            
            y_pos -= 0.2
        
        # Add trend analysis text
        trend_text = "Improving" if form.get("trajectory_score", 0) > 50 else "Stable"
        if form.get("trajectory_score", 0) < 40:
            trend_text = "Declining"
        
        ax.text(0.05, y_pos, f"Performance Trend: {trend_text}", fontsize=9, fontweight='bold')
        
        # Add recent performance note
        recent_avg = form.get("recent_average", 0)
        career_avg = self.basic_metrics.get("batting", {}).get("batting_average", 0)
        
        y_pos -= 0.1
        if recent_avg > career_avg * 1.2:
            ax.text(0.05, y_pos, f"Recent form significantly above career average", fontsize=8)
        elif recent_avg > career_avg:
            ax.text(0.05, y_pos, f"Recent form above career average", fontsize=8)
        else:
            ax.text(0.05, y_pos, f"Recent form below career average", fontsize=8)
    
    def _add_phase_analysis_panel(self, ax):
        """Add phase analysis panel to the player card."""
        ax.set_facecolor('#ffffff')
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Phase Analysis", fontsize=12, fontweight='bold', ha='center')
        
        # Get phase metrics
        phase = self.basic_metrics.get("phase", {})
        
        # Extract strike rates by phase
        pp_sr = phase.get("powerplay_strike_rate", 0)
        mid_sr = phase.get("middle_strike_rate", 0)
        death_sr = phase.get("death_strike_rate", 0)
        
        # Extract run distribution
        pp_pct = phase.get("powerplay_percentage", 0)
        mid_pct = phase.get("middle_percentage", 0)
        death_pct = phase.get("death_percentage", 0)
        
        # Create a horizontal bar chart for strike rates
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Bar chart settings
        bar_height = 0.1
        max_sr = 200  # Maximum strike rate to normalize
        
        # Draw strike rate bars
        y_pos = 0.75
        
        # Powerplay
        ax.text(0.05, y_pos, "Powerplay (1-6)", fontsize=9)
        ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9*(pp_sr/max_sr), bar_height, 
                                     facecolor='#3498db', edgecolor='none', alpha=0.8))
        ax.text(0.05 + 0.9*(pp_sr/max_sr) + 0.02, y_pos - bar_height/2, f"SR: {pp_sr:.1f}", 
               fontsize=8, va='center')
        
        # Middle overs
        y_pos -= 0.15
        ax.text(0.05, y_pos, "Middle (7-15)", fontsize=9)
        ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9*(mid_sr/max_sr), bar_height, 
                                     facecolor='#f39c12', edgecolor='none', alpha=0.8))
        ax.text(0.05 + 0.9*(mid_sr/max_sr) + 0.02, y_pos - bar_height/2, f"SR: {mid_sr:.1f}", 
               fontsize=8, va='center')
        
        # Death overs
        y_pos -= 0.15
        ax.text(0.05, y_pos, "Death (16-20)", fontsize=9)
        ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9*(death_sr/max_sr), bar_height, 
                                     facecolor='#e74c3c', edgecolor='none', alpha=0.8))
        ax.text(0.05 + 0.9*(death_sr/max_sr) + 0.02, y_pos - bar_height/2, f"SR: {death_sr:.1f}", 
               fontsize=8, va='center')
        
        # Add pie chart for run distribution
        y_pos -= 0.25
        
        # Only show pie chart if we have valid percentages
        if pp_pct + mid_pct + death_pct > 0:
            pie_size = 0.15
            pie_center = (0.3, y_pos - 0.1)
            
            # Create pie chart data
            sizes = [pp_pct, mid_pct, death_pct]
            colors = ['#3498db', '#f39c12', '#e74c3c']
            labels = ['PP', 'Mid', 'Death']
            
            # Draw pie chart manually
            start_angle = 0
            for i, (size, color, label) in enumerate(zip(sizes, colors, labels)):
                if size > 0:
                    angle = size * 3.6  # Convert percentage to degrees
                    end_angle = start_angle + angle
                    
                    # Draw wedge
                    wedge = patches.Wedge(pie_center, pie_size, start_angle, end_angle, 
                                        facecolor=color, edgecolor='white', linewidth=1)
                    ax.add_patch(wedge)
                    
                    # Calculate label position
                    label_angle = np.radians((start_angle + end_angle) / 2)
                    label_radius = pie_size * 0.7
                    label_x = pie_center[0] + np.cos(label_angle) * label_radius
                    label_y = pie_center[1] + np.sin(label_angle) * label_radius
                    
                    # Add label if slice is big enough
                    if size >= 10:
                        ax.text(label_x, label_y, f"{label}", fontsize=8, ha='center', va='center')
                    
                    start_angle = end_angle
            
            # Add title
            ax.text(pie_center[0], pie_center[1] + pie_size + 0.05, "Run Distribution", 
                   fontsize=8, ha='center')
            
            # Add legend
            legend_x = 0.5
            legend_y = y_pos
            
            for i, (color, label) in enumerate(zip(colors, ['Powerplay', 'Middle', 'Death'])):
                rect = patches.Rectangle((legend_x, legend_y - i*0.05), 0.03, 0.03, 
                                       facecolor=color, edgecolor='none')
                ax.add_patch(rect)
                ax.text(legend_x + 0.04, legend_y - i*0.05 + 0.015, f"{label}: {sizes[i]:.1f}%", 
                       fontsize=7, va='center')
    
    def _add_impact_metrics_panel(self, ax):
        """Add impact metrics panel to the player card."""
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Impact Metrics", fontsize=12, fontweight='bold', ha='center')
        
        # Get impact metrics
        impact = self.advanced_metrics.get("impact", {})
        
        # Create radar chart for impact metrics
        # In a real implementation, we would draw a proper radar chart
        # For this demo, we'll simulate it with simple bars
        
        metrics = [
            ("Batting Impact", impact.get("batting_impact_index", 0), 100),
            ("Phase Impact", impact.get("phase_impact_score", 0), 100),
            ("Boundary Impact", impact.get("boundary_impact_score", 0), 100)
        ]
        
        # Add bowling impact if applicable
        role = self.basic_metrics.get("role", "")
        if "Bowler" in role or "All-rounder" in role:
            metrics.append(("Bowling Impact", impact.get("bowling_impact_index", 0), 100))
        
        # Draw impact metrics as horizontal bars
        y_pos = 0.75
        bar_height = 0.08
        
        for label, value, max_val in metrics:
            # Normalize value
            norm_value = min(1.0, value / max_val)
            
            # Add label
            ax.text(0.05, y_pos, label, fontsize=9)
            
            # Draw background bar
            ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9, bar_height, 
                                         facecolor='#e0e0e0', edgecolor='none', alpha=0.5))
            
            # Draw value bar with color based on value
            if norm_value >= 0.8:
                color = '#2ecc71'  # Green
            elif norm_value >= 0.6:
                color = '#3498db'  # Blue
            elif norm_value >= 0.4:
                color = '#f39c12'  # Orange
            elif norm_value >= 0.2:
                color = '#e67e22'  # Dark orange
            else:
                color = '#e74c3c'  # Red
            
            ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.9*norm_value, bar_height, 
                                         facecolor=color, edgecolor='none', alpha=0.8))
            
            # Add value text
            ax.text(0.05 + 0.9*norm_value + 0.02, y_pos - bar_height/2, f"{value:.1f}", 
                   fontsize=8, va='center')
            
            y_pos -= 0.15
        
        # Add overall impact score
        overall_impact = impact.get("overall_impact_score", 0)
        
        ax.text(0.05, y_pos, "Overall Impact Score:", fontsize=10, fontweight='bold')
        y_pos -= 0.1
        
        # Draw impact score as a colored circle
        if overall_impact >= 80:
            color = '#2ecc71'  # Green
            impact_text = "Elite"
        elif overall_impact >= 65:
            color = '#3498db'  # Blue
            impact_text = "High"
        elif overall_impact >= 50:
            color = '#f39c12'  # Orange
            impact_text = "Moderate"
        elif overall_impact >= 35:
            color = '#e67e22'  # Dark orange
            impact_text = "Low"
        else:
            color = '#e74c3c'  # Red
            impact_text = "Minimal"
        
        circle = patches.Circle((0.2, y_pos), 0.08, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(0.2, y_pos, f"{int(overall_impact)}", fontsize=12, color='white', 
               ha='center', va='center', fontweight='bold')
        
        # Add impact category text
        ax.text(0.35, y_pos, f"{impact_text} Impact Player", fontsize=10, va='center')
    
    def _add_opposition_quality_panel(self, ax):
        """Add opposition quality panel to the player card."""
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Opposition Quality Analysis", fontsize=12, fontweight='bold', ha='center')
        
        # Get opposition metrics
        opposition = self.advanced_metrics.get("opposition_quality", {})
        
        # Draw bar chart for performance vs different opposition strengths
        metrics = [
            ("Strong Teams", opposition.get("avg_vs_strong_opposition", 0)),
            ("Average Teams", opposition.get("avg_vs_average_opposition", 0)),
            ("Weak Teams", opposition.get("avg_vs_weak_opposition", 0))
        ]
        
        y_pos = 0.75
        bar_height = 0.1
        max_avg = max([m[1] for m in metrics] + [50])  # At least 50 for scaling
        
        for label, value in metrics:
            # Add label
            ax.text(0.05, y_pos, label, fontsize=9)
            
            # Draw value bar
            norm_value = value / max_avg
            
            # Choose color based on opposition strength
            if label == "Strong Teams":
                color = '#3498db'  # Blue
            elif label == "Average Teams":
                color = '#f39c12'  # Orange
            else:
                color = '#e74c3c'  # Red
            
            ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.8*norm_value, bar_height, 
                                         facecolor=color, edgecolor='none', alpha=0.8))
            
            # Add value text
            ax.text(0.05 + 0.8*norm_value + 0.02, y_pos - bar_height/2, f"{value:.1f}", 
                   fontsize=8, va='center')
            
            y_pos -= 0.15
        
        # Add big match temperament score
        y_pos -= 0.05
        bmt_score = opposition.get("big_match_temperament", 0)
        ax.text(0.05, y_pos, "Big Match Temperament:", fontsize=9, fontweight='bold')
        
        y_pos -= 0.1
        # Draw as a simple gauge
        ax.add_patch(patches.Rectangle((0.05, y_pos-0.05), 0.8, 0.05, 
                                     facecolor='#e0e0e0', edgecolor='none', alpha=0.5))
        
        # Normalize value
        norm_bmt = min(1.0, bmt_score / 150)
        
        # Set color based on score
        if norm_bmt >= 0.8:
            color = '#2ecc71'  # Green
            bmt_text = "Excellent"
        elif norm_bmt >= 0.6:
            color = '#3498db'  # Blue
            bmt_text = "Good"
        elif norm_bmt >= 0.4:
            color = '#f39c12'  # Orange
            bmt_text = "Average"
        elif norm_bmt >= 0.2:
            color = '#e67e22'  # Dark orange
            bmt_text = "Below Average"
        else:
            color = '#e74c3c'  # Red
            bmt_text = "Poor"
        
        ax.add_patch(patches.Rectangle((0.05, y_pos-0.05), 0.8*norm_bmt, 0.05, 
                                     facecolor=color, edgecolor='none', alpha=0.8))
        
        # Add bmt text
        ax.text(0.05, y_pos-0.1, f"{bmt_text} ({bmt_score:.1f})", fontsize=8)
        
        # Add opposition quality score
        y_pos -= 0.15
        opp_quality_score = opposition.get("opposition_quality_score", 0)
        ax.text(0.05, y_pos, "Opposition Quality Score:", fontsize=9, fontweight='bold')
        
        y_pos -= 0.1
        # Draw as a simple gauge
        ax.add_patch(patches.Rectangle((0.05, y_pos-0.05), 0.8, 0.05, 
                                     facecolor='#e0e0e0', edgecolor='none', alpha=0.5))
        
        # Normalize value
        norm_oq = min(1.0, opp_quality_score / 100)
        
        # Draw gauge
        ax.add_patch(patches.Rectangle((0.05, y_pos-0.05), 0.8*norm_oq, 0.05, 
                                     facecolor='#2ecc71', edgecolor='none', alpha=0.8))
        
        # Add score text
        ax.text(0.05, y_pos-0.1, f"Score: {opp_quality_score:.1f}/100", fontsize=8)
    
    def _add_venue_analysis_panel(self, ax):
        """Add venue analysis panel to the player card."""
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Venue & Conditions Analysis", fontsize=12, fontweight='bold', ha='center')
        
        # Get venue metrics
        venue = self.advanced_metrics.get("venue_condition", {})
        
        # Home vs Away performance comparison
        home_avg = venue.get("home_average", 0)
        away_avg = venue.get("away_average", 0)
        home_matches = venue.get("home_matches", 0)
        away_matches = venue.get("away_matches", 0)
        
        # Draw bar chart for home vs away comparison
        y_pos = 0.75
        bar_height = 0.1
        max_avg = max(home_avg, away_avg, 30)  # At least 30 for scaling
        
        # Home performance
        ax.text(0.05, y_pos, f"Home ({home_matches} matches)", fontsize=9)
        ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.8*(home_avg/max_avg), bar_height, 
                                     facecolor='#3498db', edgecolor='none', alpha=0.8))
        ax.text(0.05 + 0.8*(home_avg/max_avg) + 0.02, y_pos - bar_height/2, f"{home_avg:.1f}", 
               fontsize=8, va='center')
        
        # Away performance
        y_pos -= 0.15
        ax.text(0.05, y_pos, f"Away ({away_matches} matches)", fontsize=9)
        ax.add_patch(patches.Rectangle((0.05, y_pos-bar_height), 0.8*(away_avg/max_avg), bar_height, 
                                     facecolor='#e74c3c', edgecolor='none', alpha=0.8))
        ax.text(0.05 + 0.8*(away_avg/max_avg) + 0.02, y_pos - bar_height/2, f"{away_avg:.1f}", 
               fontsize=8, va='center')
        
        # Venue adaptability score
        y_pos -= 0.15
        adaptability = venue.get("venue_adaptability_score", 0)
        ax.text(0.05, y_pos, "Venue Adaptability:", fontsize=9, fontweight='bold')
        ax.text(0.05, y_pos-0.1, f"Score: {adaptability:.1f}/100", fontsize=9)
    
    def _add_breakout_potential_panel(self, ax):
        """Add breakout potential panel to the player card."""
        ax.set_facecolor('#f5f5f5')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Panel title
        ax.text(0.5, 0.9, "Breakout Potential Assessment", fontsize=14, fontweight='bold', ha='center')
        
        # Get breakout potential metrics
        breakout = self.advanced_metrics.get("breakout_potential", {})
        score = breakout.get("breakout_potential_score", 0)
        category = breakout.get("breakout_category", "N/A")
        
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
        
        # Draw large score circle
        circle = patches.Circle((0.2, 0.6), 0.12, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.2, 0.6, f"{int(score)}", fontsize=18, color='white', 
               ha='center', va='center', fontweight='bold')
        
        # Add category
        ax.text(0.2, 0.4, f"{category}", fontsize=14, 
               ha='center', va='center', fontweight='bold')
        ax.text(0.2, 0.32, "Breakout Potential", fontsize=10, 
               ha='center', va='center')
        
        # Add key strengths
        key_strengths = breakout.get("key_strengths", [])
        if key_strengths:
            ax.text(0.45, 0.75, "Key Strengths:", fontsize=12, fontweight='bold')
            for i, strength in enumerate(key_strengths[:3]):  # Only show top 3
                ax.text(0.48, 0.7 - i*0.08, f"• {strength}", fontsize=10)
        
        # Add areas for improvement
        areas_for_improvement = breakout.get("areas_for_improvement", [])
        if areas_for_improvement:
            ax.text(0.45, 0.45, "Areas for Improvement:", fontsize=12, fontweight='bold')
            for i, area in enumerate(areas_for_improvement[:2]):  # Only show top 2
                ax.text(0.48, 0.4 - i*0.08, f"• {area}", fontsize=10)
        
        # Add conclusion statement
        if score >= 80:
            conclusion = "Elite breakout potential. Strong performer across all metrics."
        elif score >= 65:
            conclusion = "High breakout potential. Demonstrates strong ability in key areas."
        elif score >= 50:
            conclusion = "Moderate breakout potential. Shows promise but needs consistency."
        elif score >= 35:
            conclusion = "Limited breakout potential. Significant improvement needed."
        else:
            conclusion = "Very limited breakout potential. Major development required."
        
        ax.text(0.5, 0.15, "Conclusion:", fontsize=12, fontweight='bold', ha='center')
        ax.text(0.5, 0.08, conclusion, fontsize=10, ha='center')
    
    def save_player_card(self, fig):
        """Save the player card visualization to a file."""
        filename = os.path.join(VISUALIZATION_DIR, f"{self.player_name.replace(' ', '_')}_player_card.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved player card for {self.player_name} to {filename}")