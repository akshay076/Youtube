# Delhi Capitals Breakout Player Analysis

A modular Python project for analyzing potential breakout players for the Delhi Capitals IPL team.

## Project Overview

This project implements a comprehensive methodology to identify and analyze potential breakout players for Delhi Capitals, focusing on players with limited IPL experience (≤20 matches). The analysis combines traditional cricket statistics with advanced metrics to assess form, opposition quality, venue adaptability, and overall impact.

## Key Features

- **Data Collection**: Fetches and processes player statistics
- **Basic Metrics Analysis**: Calculates fundamental cricket performance indicators
- **Advanced Metrics Analysis**: Implements sophisticated metrics for breakout prediction
- **Player Card Visualization**: Creates comprehensive visual player cards
- **Comparison Visualizations**: Generates comparison charts across various metrics

## Project Structure

```
delhi_capitals_analysis/
├── config/
│   └── config.py             # Configuration settings
├── data/                     # Data storage
│   └── raw/                  # Raw collected data
├── src/                      # Source code
│   ├── data/                 
│   │   └── data_collector.py # Data collection module
│   ├── analysis/
│   │   ├── basic_metrics.py  # Basic statistical metrics
│   │   └── advanced_metrics.py # Advanced analytical metrics
│   └── visualization/
│       ├── player_cards.py   # Player card visualizations
│       └── comparison_plots.py # Comparative analysis plots
├── output/                   # Generated analysis and visualizations
│   ├── metrics/              # Calculated metrics
│   └── visualizations/       # Generated plots and charts
├── main.py                   # Main execution script
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/delhi-capitals-analysis.git
   cd delhi-capitals-analysis
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Complete Pipeline

To run the complete analysis pipeline for all potential breakout players:

```
python main.py --all
```

### Running Individual Steps

To collect data only:
```
python main.py --collect
```

To analyze previously collected data:
```
python main.py --analyze
```

To create visualizations from analysis results:
```
python main.py --visualize
```

### Analyzing Specific Players

To analyze specific players:
```
python main.py --all --players "Jake Fraser-McGurk" "Tristan Stubbs" "Abishek Porel"
```

## Analysis Methodology

The analysis methodology uses several key components:

1. **Form-Based Metrics**
   - Recent Performance Weighting
   - Performance Trajectory Analysis
   - Consistency Assessment

2. **Opposition Quality Adjustment**
   - Performance vs. Strong Teams
   - Big Match Temperament

3. **Phase-Specific Analysis**
   - Performance in Powerplay/Middle/Death Overs

4. **Venue and Condition Analysis**
   - Home vs. Away Performance
   - Venue Adaptability

5. **Impact Assessment**
   - Batting Impact Index
   - Phase Impact Score
   - Overall Impact Calculation

6. **Breakout Prediction**
   - Combined Breakout Potential Score
   - Key Strengths and Areas for Improvement

## Output Examples

The analysis produces various outputs:

1. **Individual Player Cards**: Comprehensive visualization of a player's metrics, strengths, and breakout potential

2. **Comparison Charts**:
   - Breakout Potential Comparison
   - Impact Metrics Comparison
   - Form and Trajectory Analysis
   - Top Breakout Candidates Summary

## Notes

- This implementation uses synthetic data generation for demonstration purposes
- In a real-world scenario, data would be collected from cricket APIs or websites

## Future Enhancements

- Implement real data collection from cricket statistics APIs
- Add machine learning models for performance prediction
- Expand analysis to bowling-specific metrics
- Create interactive dashboards with historical trends

## License

This project is licensed under the MIT License - see the LICENSE file for details.